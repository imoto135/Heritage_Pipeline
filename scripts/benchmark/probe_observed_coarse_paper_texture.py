"""Add only observed coarse L variation, keeping component mean and fine texture."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_preserved_paper_boundary as prior
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT/'tmp_work/observed_coarse_paper_texture_pilot_20261001'
SIGMAS = [24, 48]
SETTINGS = dict(baseline='saved thin+8px fragment original16-pair paper fill; no preserved-color correction',
    mask_applied_color_reference_and_fine_texture_fixed=True,
    source_coordinates='same saved fine texture source x/y, every source inside original known paper',
    coarse_band='known-paper normalized Gaussian L sigma6 minus sigma24 or sigma48',
    coarse_source_clip_L=2, target_smoothing='normalized Gaussian sigma6 over transferred observed band only',
    mean='subtract feather-weighted component mean, then linearly shrink only if amplitude exceeds2L',
    boundary='existing texture feather distance1 zero, distance5 full',
    preserved_fields='a/b and each applied component mean L; RGB rounding/clipping may slightly shift means',
    nonreference_signal=False, noise=False, ROIs=42, development=True, pixel_GT=False, DDRM_recomputed=False)


def known_lowpass(light, known, sigma):
    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    return cv2.GaussianBlur(np.where(known, light, 0), (0, 0), sigma)/np.maximum(mass, 1e-6)


def coarse_transfer(light, known, applied, sx, sy, sigma):
    selected = (sx >= 0)&(sy >= 0); assert not np.any(selected&~applied)
    assert known[sy[selected], sx[selected]].all()
    low6 = known_lowpass(light, known, 6); low = known_lowpass(light, known, sigma)
    band = low6-low; borrowed = np.zeros_like(light)
    borrowed[selected] = np.clip(band[sy[selected], sx[selected]], -2, 2)
    mass = cv2.GaussianBlur(selected.astype(np.float32), (0, 0), 6)
    smoothed = cv2.GaussianBlur(borrowed, (0, 0), 6)/np.maximum(mass, 1e-6)
    distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((distance-1)/4., 0, 1)*selected
    correction = np.zeros_like(light); notes = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(applied.astype(np.uint8), connectivity=4)
    for i in range(1, n):
        target = labels == i; ww = weight[target].astype(np.float64); values = smoothed[target].astype(np.float64)
        if ww.sum() > 0:
            center = float(np.sum(values*ww)/ww.sum()); delta = (values-center)*ww
            maximum = float(np.abs(delta).max()); scale = min(1., 2./max(maximum, 1e-6))
            correction[target] = (delta*scale).astype(np.float32)
            assert abs(float(correction[target].mean(dtype=np.float64))) < 1e-5
            notes.append(dict(id=i, area=int(stats[i, 4]), source_pixels=int(selected[target].sum()),
                center=center, scale=scale, mean_correction=float(correction[target].mean(dtype=np.float64)),
                std_correction=float(correction[target].std()), max_abs=float(np.abs(correction[target]).max())))
    assert not correction[~applied].any() and float(np.abs(correction).max()) <= 2+1e-6
    assert not correction[distance <= 1].any()
    return correction, borrowed, smoothed, notes


def main():
    for folder in [prior.OUT, prior.BASE]: check_bundle(folder)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((prior.OUT/'diagnostics.json').read_text()); pages = []
    for source in prior.references.prior.prior.mask.prior.MARGIN.SOURCES+[prior.references.prior.prior.mask.new.INPUT]:
        pages += json.loads((source/'baseline_review.json').read_text())['cases']
    textures = json.loads((prior.OUT/'texture_sources.json').read_text())
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        **{str((folder/name).relative_to(ROOT)): sha(folder/name) for folder in [prior.OUT, prior.BASE] for name in ['manifest.json', 'complete.json']},
        **{p['original']: sha(ROOT/p['original']) for p in pages}}
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py'); results = []; poison = []
    for pg in pages:
        key = pg['page']; original = read(ROOT/pg['original'])
        for row in [r for r in rows if r['page'] == key]:
            rid = row['id']; x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
            context = original[cy:cy+ch, cx:cx+cw]; s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w)); crop = context[s]
            lab = cv2.cvtColor(context.astype(np.float32)/255, cv2.COLOR_BGR2LAB)
            native = prior.OUT/'native'/rid; dest = OUT/'native'/rid
            base = read(native/'base.png'); applied = read(native/'applied_context.png', True)
            known = read(native/'baseline_reference_context.png', True); outputs = {}; diagnostics = {}
            if applied[s].any():
                source = ROOT/textures[rid]; sx = np.load(source/'source_x.npy', allow_pickle=False); sy = np.load(source/'source_y.npy', allow_pickle=False)
                base_field = np.load(native/'base_field_Lab.npy', allow_pickle=False)
            for sigma in SIGMAS:
                mode = 'coarse'+str(sigma); md = dest/mode; value = base.copy(); notes = []
                if applied[s].any():
                    delta, borrowed, smoothed, notes = coarse_transfer(lab[:, :, 0], known, applied, sx, sy, sigma)
                    field = base_field.copy(); field[:, :, 0] += delta
                    assert np.array_equal(field[:, :, 1:], base_field[:, :, 1:])
                    value = prior.references.prior.prior.fill.frozen.compose(field, applied, base, s)
                    poisoned = lab[:, :, 0].copy(); poisoned[~known] = -250
                    repeat = coarse_transfer(poisoned, known, applied, sx, sy, sigma)
                    assert np.array_equal(delta, repeat[0]) and np.array_equal(borrowed, repeat[1]); poison.append(dict(id=rid, sigma=sigma))
                    md.mkdir(parents=True, exist_ok=True)
                    for name, item in dict(coarse_delta_L=delta, borrowed_L=borrowed, smoothed_L=smoothed, result_field_Lab=field).items():
                        np.save(md/(name+'.npy'), item, allow_pickle=False)
                assert not np.any(np.any(value != base, axis=2)&~applied[s])
                outputs[mode] = value; diagnostics[mode] = dict(changed_vs_base=int(np.any(value != base, axis=2).sum()), component_notes=notes)
                write(md/'output.png', value); dump(md/'components.json', notes)
            for name, item in dict(original=crop, base=base, applied_context=applied, reference_context=known).items(): write(dest/(name+'.png'), item)
            figure([crop, base]+[outputs['coarse'+str(v)] for v in SIGMAS],
                ['原画像', '前回・細かい模様', '平均色固定・模様6..24px', '平均色固定・模様6..48px'],
                OUT/'comparisons'/(rid+'.png'), row['title']+' · 観測された粗い明度模様だけ',
                '各成分の平均L・a/b・細かい模様・マスクを固定。粗い補正は±2L以内。',
                '同じ参照元座標の既知紙面の明度差分を移す。原損傷・DDRM・乱数を使わない。')
            results.append(dict(**row, coarse_texture_diagnostics=diagnostics)); print('Observed coarse L:', len(results), '/42', rid, flush=True)
        del original; gc.collect()
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', results)
    dump(OUT/'verification.json', dict(status='passed', ROIs=42, variants=84, mask_applied_fine_texture_fixed=True,
        mean_L_per_component_preserved=True, ab_field_identical=True, first_inside_correction_zero=True,
        max_L_correction=2, observed_source_known_only=True, all_nonreference_poison_cases=poison,
        outside_applied_changes=0, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: mean-preserving observed coarse L only, 84 variants', flush=True)


if __name__ == '__main__': main()
