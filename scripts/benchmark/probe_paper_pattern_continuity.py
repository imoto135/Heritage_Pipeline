"""Fixed-mask paper continuity diagnostic using exclusively known paper donors."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import probe_adjacent_paper_boundary as prior
from build_npj_v6_comparison import dump, read, sha, write

ROOT = prior.ROOT
OUT = ROOT / 'tmp_work/paper_pattern_continuity_pilot_20261001'
MODES = ['structure', 'texture', 'structure_texture']
SETTINGS = dict(mask_fixed=True, DDRM_fixed=True, baseline='saved adjacent boundary-only output',
    scope='same prior adjusted mask; no new edit pixels', context_px=64,
    donor_patch_px=24, donor_stride_px=8, donor_max_count=512, minimum_donors=4,
    tile_overlap_px=8, overlap_cost='mean squared Lab highpass residual plus .02 times donor reuse count',
    donor_gate='all 24x24 pixels must belong to saved unedited paper reference',
    reference_lowpass_sigma_px=6, texture_lowpass_sigma_px=3,
    structure='screened harmonic Lab field from known-paper lowpass boundary; retain generated highpass',
    screening_weight=1/48**2, texture='overlap-weighted observed-paper highpass tiles; retain generated lowpass',
    inward_feather_px=4, original_damage_blending=False, generated_random_noise=False,
    pixel_GT=False, development_regions=22, training=False, inference=False)


def smooth_known(lab, known, sigma):
    weight = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    value = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), sigma)
    return value / np.maximum(weight[:, :, None], 1e-6)


def donors_from_known(lab, known):
    k = SETTINGS['donor_patch_px']; positions = []
    h, w = known.shape
    integral = cv2.integral(known.astype(np.uint8))
    for y in range(0, h-k+1, 8):
        for x in range(0, w-k+1, 8):
            count = integral[y+k, x+k]-integral[y, x+k]-integral[y+k, x]+integral[y, x]
            if count == k*k:
                positions.append((x, y))
    if len(positions) > 512:
        positions = [positions[i] for i in np.linspace(0, len(positions)-1, 512, dtype=int)]
    if len(positions) < 4:
        return None, positions
    residual = lab - smooth_known(lab, known, 3)
    patches = np.stack([residual[y:y+k, x:x+k].copy() for x, y in positions])
    return patches, positions


def tile_observed_pattern(patches, shape):
    h, w = shape; k = 24; stride = 16
    H, W = max(h, k), max(w, k)
    total = np.zeros((H+k, W+k, 3), np.float32)
    weight = np.zeros((H+k, W+k), np.float32)
    use = np.zeros(len(patches), np.int32)
    axis = np.minimum(1, np.minimum(np.arange(k)+1, k-np.arange(k))/8.)
    tile_weight = (axis[:, None]*axis[None, :]).astype(np.float32)
    first = int(np.argsort(np.std(patches[:, :, :, 0], axis=(1, 2)))[len(patches)//2])
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            old_weight = weight[y:y+k, x:x+k]
            overlap = old_weight > 0
            if overlap.any():
                old = total[y:y+k, x:x+k] / np.maximum(old_weight[:, :, None], 1e-6)
                cost = np.mean((patches[:, overlap]-old[overlap])**2, axis=(1, 2)) + .02*use
                chosen = int(np.argmin(cost))
            else:
                chosen = first
            total[y:y+k, x:x+k] += patches[chosen]*tile_weight[:, :, None]
            weight[y:y+k, x:x+k] += tile_weight
            use[chosen] += 1
    texture = total[:h, :w] / np.maximum(weight[:h, :w, None], 1e-6)
    return texture, dict(tile_count=int(use.sum()), distinct_donors=int((use>0).sum()),
                        most_reused_donor_tiles=int(use.max()))


def harmonic_field(component, known, reference_lowpass, generated_lowpass):
    yy, xx = np.nonzero(component); count = len(yy)
    index = np.full(component.shape, -1, np.int32); index[yy, xx] = np.arange(count)
    diagonal = np.full(count, 1/48**2, np.float64)
    rhs = generated_lowpass[yy, xx].astype(np.float64) / 48**2
    row, col, entries = [], [], []
    h, w = component.shape; boundary_pairs = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = yy+dy, xx+dx
        valid = (ny>=0)&(ny<h)&(nx>=0)&(nx<w)
        ii = np.flatnonzero(valid); oy, ox = ny[valid], nx[valid]
        inside = component[oy, ox]
        ids = ii[inside]; diagonal[ids] += 1
        row.extend(ids.tolist()); col.extend(index[oy[inside], ox[inside]].tolist())
        entries.extend([-1.]*len(ids))
        boundary = (~inside)&known[oy, ox]
        ids = ii[boundary]; diagonal[ids] += 1
        rhs[ids] += reference_lowpass[oy[boundary], ox[boundary]]
        boundary_pairs += len(ids)
    if boundary_pairs < 16:
        return None, boundary_pairs
    row.extend(range(count)); col.extend(range(count)); entries.extend(diagonal.tolist())
    matrix = sparse.coo_matrix((entries, (row, col)), shape=(count, count)).tocsr()
    result = generated_lowpass.copy(); result[yy, xx] = spsolve(matrix, rhs).astype(np.float32)
    assert np.isfinite(result).all()
    return result, boundary_pairs


def infer(original_context, reference, baseline, active, offset):
    ox, oy = offset; h, w = baseline.shape[:2]
    original_lab = prior.previous.to_lab(original_context)
    base_lab = prior.previous.to_lab(baseline)
    local_ref = reference[oy:oy+h, ox:ox+w]
    paper = smooth_known(original_lab, reference, 6)[oy:oy+h, ox:ox+w]
    patches, positions = donors_from_known(original_lab, reference)
    values = {m: base_lab.copy() for m in MODES}; target = np.zeros((h, w), bool)
    notes = []; donor_used = np.zeros(reference.shape, bool)
    if patches is None:
        return {m: baseline.copy() for m in MODES}, target, donor_used, positions, notes
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), connectivity=8)
    for i in range(1, count):
        x, y, rw, rh, area = map(int, stats[i])
        if area < 64:
            continue
        component = labels == i
        generated_low = smooth_known(base_lab, component, 3)
        field, pairs = harmonic_field(component, local_ref, paper, generated_low)
        if field is None:
            notes.append(dict(id=i, area=area, applied=False, boundary_pairs=pairs)); continue
        residual, tile_notes = tile_observed_pattern(patches, (rh, rw))
        observed = np.zeros_like(base_lab); observed[y:y+rh, x:x+rw] = residual
        distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        feather = np.clip(distance/4, 0, 1)[:, :, None]
        candidates = dict(structure=field+base_lab-generated_low,
                          texture=generated_low+observed, structure_texture=field+observed)
        for mode, candidate in candidates.items():
            value = base_lab + feather*(candidate-base_lab)
            values[mode][component] = value[component]
        target |= component
        notes.append(dict(id=i, area=area, applied=True, boundary_pairs=pairs, **tile_notes))
    for x, y in positions:
        donor_used[y:y+24, x:x+24] = True
    outputs = {}
    for mode, value in values.items():
        image = baseline.copy(); image[target] = prior.previous.from_lab(value)[target]
        outputs[mode] = image
    return outputs, target, donor_used, positions, notes


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                    str(Path(prior.__file__).relative_to(ROOT)): sha(Path(prior.__file__)),
                    str(Path(prior.previous.__file__).relative_to(ROOT)): sha(Path(prior.previous.__file__))}
    for folder in [prior.OUT, prior.previous.OUT, prior.DDRM]:
        for name in ['manifest.json', 'complete.json']:
            dependencies[str((folder/name).relative_to(ROOT))] = sha(folder/name)
        for rel, digest in json.loads((folder/'manifest.json').read_text())['dependencies'].items():
            assert sha(ROOT/rel) == digest, rel
        for rel, digest in json.loads((folder/'complete.json').read_text())['outputs'].items():
            assert sha(folder/rel) == digest, rel
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)),
                    pages=14, regions=22, selection='all 22 prior development ROIs; no new selection')
    if (OUT/'manifest.json').exists():
        assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else:
        dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    rows = []; poison_checked = []
    for source in prior.previous.margin_probe.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            original = read(ROOT/page['original']); H, W = original.shape[:2]
            full_edit = read(prior.MASKS/'candidate_pages'/page['page']/'dark_support/edit.png', True)
            for roi in page['rois']:
                key = roi['id']; x, y, w, h = roi['xywh']
                x0, y0 = max(0, x-64), max(0, y-64); x1, y1 = min(W, x+w+64), min(H, y+h+64)
                context = original[y0:y1, x0:x1]; local = (slice(y-y0, y-y0+h), slice(x-x0, x-x0+w))
                native = prior.OUT/'native'/key
                base = read(native/'boundary.png'); reference = read(native/'reference_context.png', True)
                active = read(native/'adjusted.png', True); edit = full_edit[y:y+h, x:x+w]
                assert not np.any(reference & full_edit[y0:y1, x0:x1])
                assert np.array_equal(read(native/'original.png'), context[local])
                if active.any():
                    outputs, target, used, positions, notes = infer(context, reference, base, active, (x-x0, y-y0))
                    if key in ['hole_edge_new', 'new_margin_hole', 'new_brown_stamp']:
                        poisoned = context.copy(); poisoned[full_edit[y0:y1, x0:x1]] = (217, 7, 103)
                        repeat = infer(poisoned, reference, base, active, (x-x0, y-y0))[0]
                        for mode in MODES:
                            assert np.array_equal(outputs[mode], repeat[mode])
                        poison_checked.append(key)
                else:
                    outputs = {m: base.copy() for m in MODES}; target = active.copy()
                    used = np.zeros(reference.shape, bool); positions = []; notes = []
                assert not np.any(used & ~reference)
                assert not np.any(target & ~active)
                dest = OUT/'native'/key
                maps = dict(original=context[local], base=base, edit=edit, adjusted=target,
                            donor_context=used, reference_context=reference, **outputs)
                for name, value in maps.items():
                    write(dest/(name+'.png'), value)
                    assert np.array_equal(read(dest/(name+'.png'), value.dtype==bool), value)
                dump(dest/'donors.json', dict(context_xywh=[x0,y0,x1-x0,y1-y0], patch_xy=positions, components=notes))
                metrics = {}
                evaluation = read(prior.previous.OUT/'native'/key/'adjusted.png', True)
                for mode, image in dict(base=base, **outputs).items():
                    assert not np.any(np.any(image!=base, axis=2) & ~target)
                    assert not np.any(np.any(image!=context[local], axis=2) & ~edit)
                    lab = prior.previous.to_lab(image); jumps = []
                    ref = reference[local]
                    for a, b, ra, rb, delta in [
                        (evaluation[:-1], evaluation[1:], ref[:-1], ref[1:], lab[:-1]-lab[1:]),
                        (evaluation[:,:-1], evaluation[:,1:], ref[:,:-1], ref[:,1:], lab[:,:-1]-lab[:,1:])]:
                        valid = (a & rb) | (b & ra)
                        if valid.any(): jumps.append(np.linalg.norm(delta[valid], axis=1))
                    values = np.concatenate(jumps) if jumps else np.empty(0)
                    distance = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 5)
                    core = target & (distance>=6)
                    high = lab[:,:,0]-cv2.GaussianBlur(lab[:,:,0], (0,0), 1.2)
                    mid = cv2.GaussianBlur(lab[:,:,0], (0,0), 2)-cv2.GaussianBlur(lab[:,:,0], (0,0), 8)
                    metrics[mode] = dict(boundary_pairs=len(values),
                        median_boundary_deltaLab=float(np.median(values)) if len(values) else None,
                        core_highpass_std=float(np.std(high[core])) if core.any() else None,
                        core_midband_std=float(np.std(mid[core])) if core.any() else None,
                        changed_vs_base=int(np.any(image!=base, axis=2).sum()))
                prior.previous.figure([context[local], base, outputs['structure'], outputs['texture'], outputs['structure_texture']],
                    ['原画像','前回の境界補正','紙色のつながり','紙模様のみ','紙色＋紙模様'],
                    OUT/'comparisons'/(key+'.png'), roi['title']+' · 紙面の連続性',
                    f'対象 {int(target.sum()):,}画素。未編集紙面の24px角パッチ {len(positions)}個から模様を取得。')
                contour = context[local].copy()
                contours, _ = cv2.findContours(edit.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour, contours, -1, (0,0,255), 1)
                prior.previous.figure([context[local], contour, base, outputs['structure_texture']],
                    ['原画像','赤線＝現在の編集境界','前回の境界補正','紙色＋紙模様'],
                    OUT/'mask_diagnostic'/(key+'.png'), roi['title']+' · マスク外と内部の切り分け',
                    '赤線は表示用。今回マスク外の原画素と残存損傷は変わりません。')
                rows.append(dict(id=key, page=page['page'], title=roi['title'], xywh=roi['xywh'],
                                 adjusted_pixels=int(target.sum()), donor_patches=len(positions), components=notes, metrics=metrics))
                print('Paper pattern:', len(rows), '/22', key, 'pixels', int(target.sum()), flush=True)
            del original, full_edit; gc.collect()
    dump(OUT/'diagnostics.json', rows)
    for rel, digest in dependencies.items(): assert sha(ROOT/rel)==digest, rel
    dump(OUT/'verification.json', dict(status='passed', regions=22, pages=14, mask_and_DDRM_fixed=True,
        outside_edit_changed_pixels=0, outside_adjusted_changed_pixels=0, donor_entire_patch_is_known_paper=True,
        original_damage_poison_invariance=poison_checked, generated_noise=False, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json'] and p.name not in ['complete.json','manifest.json']}))
    print('Complete: paper pattern diagnostic, all 22 ROIs', flush=True)


if __name__ == '__main__':
    main()
