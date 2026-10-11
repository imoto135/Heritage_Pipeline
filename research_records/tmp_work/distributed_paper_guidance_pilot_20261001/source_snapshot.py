"""Fixed masks and texture; screened harmonic tone from distributed known paper."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import probe_paper_boundary_pair_gate_v2 as prior
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/distributed_paper_guidance_pilot_20261001'
SIGMAS = [16, 32]
SETTINGS = dict(mask_reference_texture_fixed=True, added_components_only=True,
    old_applied_pixels_preserved=True, sigmas_px=SIGMAS, screening_strength=1/256,
    screening_mass='normalized known-paper Gaussian mass; zero below .05',
    operator='4-neighbor harmonic plus mass-weighted distributed paper Lab guidance',
    boundary='same observed raw and reference-only sigma6 smooth values',
    texture='exact saved seam; raw/smooth distance blending unchanged',
    no_damage_or_DDRM_guidance=True, pages=18, ROIs=34,
    development=True, pixel_GT=False, DDRM_recomputed=False)


def solve_screened(lab, known, applied, sigma):
    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    guide = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), sigma)/np.maximum(mass[:, :, None], 1e-6)
    smooth = prior.reference_lowpass(lab, known)
    strength = np.where(mass >= .05, mass/256., 0).astype(np.float32)*applied
    fields = {name: np.zeros_like(lab) for name in ['boundary_raw', 'boundary_smooth']}
    used = np.zeros_like(known); notes = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(applied.astype(np.uint8), connectivity=4)
    H, W = applied.shape
    for k in range(1, n):
        component = labels == k; yy, xx = np.nonzero(component); N = len(yy)
        index = np.full(applied.shape, -1, np.int32); index[yy, xx] = np.arange(N)
        coefficient = strength[yy, xx].astype(np.float64)
        diagonal = coefficient.copy()
        rhs = coefficient[:, None]*np.tile(guide[yy, xx], (1, 2))
        rows = []; cols = []; values = []; pairs = 0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = yy+dy, xx+dx; valid = (ny >= 0)&(ny < H)&(nx >= 0)&(nx < W)
            ids = np.flatnonzero(valid); ny, nx = ny[valid], nx[valid]
            inside = component[ny, nx]; ii = ids[inside]; diagonal[ii] += 1
            rows.extend(ii.tolist()); cols.extend(index[ny[inside], nx[inside]].tolist()); values.extend([-1.]*len(ii))
            boundary = known[ny, nx]; ii = ids[boundary]; diagonal[ii] += 1
            rhs[ii] += np.concatenate([lab[ny[boundary], nx[boundary]], smooth[ny[boundary], nx[boundary]]], axis=1)
            used[ny[boundary], nx[boundary]] = True; pairs += len(ii)
        assert pairs >= 8
        rows.extend(range(N)); cols.extend(range(N)); values.extend(diagonal.tolist())
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=(N, N)).tocsr()
        solution = spsolve(matrix, rhs); residual = float(np.abs(matrix@solution-rhs).max())
        assert np.isfinite(solution).all() and residual < 1e-6
        for j, name in enumerate(fields): fields[name][yy, xx] = solution[:, j*3:j*3+3].astype(np.float32)
        notes.append(dict(id=k, xywh=stats[k, :4].tolist(), area=N, pairs=pairs, residual=residual,
            guide_L_min=float(guide[component, 0].min()), guide_L_max=float(guide[component, 0].max()),
            mass_min=float(mass[component].min()), mass_max=float(mass[component].max())))
    return fields, guide, mass, strength, used, notes


def field_with_texture(fields, applied, seam):
    distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
    bw = np.exp(-(np.maximum(distance-1, 0)/8.)**2)*applied
    result = fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:, :, None]
    result[:, :, 0] += seam*np.clip((distance-1)/4., 0, 1)*applied
    return result


def main():
    # Basic well-posedness and no-reference abstention before freezing a run.
    lab = np.full((32, 32, 3), (90, 3, 12), np.float32)
    mask = np.zeros((32, 32), bool); mask[10:22, 10:22] = True; lab[mask] = 0
    for sigma in SIGMAS:
        fields, *_ = solve_screened(lab, ~mask, mask, sigma)
        for field in fields.values(): assert np.allclose(field[mask], (90, 3, 12), atol=1e-4)
    OUT.mkdir(parents=True, exist_ok=True); check_bundle(prior.OUT)
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        **{str((prior.OUT/name).relative_to(ROOT)): sha(prior.OUT/name) for name in ['manifest.json', 'complete.json']}}
    rows = json.loads((prior.OUT/'diagnostics.json').read_text())
    originals = {}
    for source in prior.frozen.MARGIN.SOURCES + [prior.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            originals[page['page']] = page['original']
            dependencies[page['original']] = sha(ROOT/page['original'])
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    results = []; poison = []; current = None
    for row in rows:
        rid = row['id']; key = row['page']
        if current != key:
            original = read(ROOT/originals[key]); current = key; gc.collect()
        cx, cy, cw, ch = row['context_xywh']; x, y, w, h = row['xywh']
        context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
        native = prior.OUT/'native'/rid; dest = OUT/'native'/rid
        known = read(native/'reference_context.png', True); applied = read(native/'applied8_context.png', True)
        extra = read(native/'newly_applied_context.png', True); base = read(native/'pairs8.png')
        crop = context[s]; outputs = {}; diagnostics = {}
        lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
        for sigma in SIGMAS:
            mode = 'guide'+str(sigma); value = base.copy()
            if extra.any():
                fields, guide, mass, strength, used, notes = solve_screened(lab, known, applied, sigma)
                seam = np.load(native/'texture_seam.npy', allow_pickle=False)
                field = field_with_texture(fields, applied, seam)
                value = prior.frozen.compose(field, extra, base, s)
                if extra[s].any():
                    poisoned = lab.copy(); poisoned[~known] = (-300, 200, -200)
                    repeat = solve_screened(poisoned, known, applied, sigma)
                    for name in fields: assert np.array_equal(fields[name], repeat[0][name])
                    assert np.array_equal(guide, repeat[1]); poison.append(dict(id=rid, sigma=sigma))
                md = dest/mode; md.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(md/'screened_fields_Lab.npz', **fields)
                for name, value_array in dict(guide_Lab=guide, reference_mass=mass, screening_strength=strength, result_field_Lab=field).items():
                    np.save(md/(name+'.npy'), value_array, allow_pickle=False)
                write(md/'used_context.png', used); dump(md/'components.json', notes)
            assert not np.any(np.any(value != base, axis=2)&~extra[s])
            local_lab = cv2.cvtColor(value.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            outputs[mode] = value
            diagnostics[mode] = dict(changed_vs_pairs8=int(np.any(value != base, axis=2).sum()),
                median_L=float(np.median(local_lab[extra[s], 0])) if extra[s].any() else None)
        for name, value in dict(original=crop, base=base, newly_applied=extra[s], **outputs).items(): write(dest/(name+'.png'), value)
        figure([crop, base]+[outputs['guide'+str(s)] for s in SIGMAS],
            ['原画像', '隣接境界のみ・8組', '分布参照 σ16', '分布参照 σ32'], OUT/'comparisons'/(rid+'.png'),
            row['title']+' · 紙色参照の位置だけを拡張',
            f'固定の新規補間範囲 {int(extra[s].sum()):,}画素。原損傷の色は混ぜない。',
            'マスク・参照資格・模様を固定。隣接境界と周囲の観測紙色で補間。DDRM再推論なし。')
        results.append(dict(**row, diagnostics=diagnostics))
        print('Distributed paper guidance:', len(results), '/34', rid, flush=True)
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', results)
    dump(OUT/'verification.json', dict(status='passed', constant_paper=True,
        all_nonreference_poison_cases=poison, outside_newly_applied_changes=0,
        masks_reference_texture_fixed=True, previous_applied_unchanged=True, ROIs=34, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.npz', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: distributed observed-paper tone, 34 ROIs', flush=True)


if __name__ == '__main__': main()
