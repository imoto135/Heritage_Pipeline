"""Compare paper colour after excluding a numerical band of nearby references.

The enlarged unknown set is used only in the colour solver. Actual image edits,
directional holds and the saved observed-L texture remain fixed.
"""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_raw_qualified_paper_colour as prior
from probe_observed_absolute_paper_patches import metrics
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT / 'tmp_work/paper_reference_setback_pilot_20261001'
RADII = [0, 3, 6]
SETTINGS = dict(
    pages=29, ROIs=90, radii_px=RADII,
    baseline='directional abstention77 + new4-books13; saved colour and observed L',
    fixed='image edit, effective applied, directional holds, original texture and qualification',
    virtual_unknown='applied OR (original known paper AND distance to applied <= radius)',
    reference='original known paper minus virtual unknown; no qualification relaxation',
    distance='OpenCV Euclidean distance transform mask5, on the saved 64px context',
    solve='same4-neighbour raw/sigma6 harmonic Lab, minimum area64 and boundary16 pairs',
    colour_blend='original applied distance exp(-(max(distance-1,0)/8)^2), not virtual distance',
    image_update='supported virtual unknown intersect original applied only',
    fallback='retain original saved field and output for unsupported applied pixels',
    texture='saved baseline float field minus saved baseline colour, never recomputed',
    control='radius0 must exactly reproduce all90 saved baseline images',
    no_new_applied=True, patches_combined=False, raw_gate_combined=False,
    texture_energy_combined=False, new_DDRM=False, pixel_GT=False, development=True)


def geometry(app, known, radius):
    assert not np.any(app & known)
    distance = cv2.distanceTransform((~app).astype(np.uint8), cv2.DIST_L2, 5)
    band = known & (distance <= radius) if radius else np.zeros_like(known)
    active = app | band
    return active, known & ~active, band


def infer(lab, known, app, field, basecolour, radius):
    active, reference, band = geometry(app, known, radius)
    fields, supported, used, notes = prior.fill.solve_fields(lab, reference, active, 16)
    actual = app & supported
    distance = cv2.distanceTransform(app.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.exp(-(np.maximum(distance - 1, 0) / 8.) ** 2) * app
    colour = fields['boundary_smooth'] + (fields['boundary_raw'] - fields['boundary_smooth']) * weight[:, :, None]
    delta = np.zeros_like(field)
    delta[actual] = colour[actual] - basecolour[actual]
    result = field + delta
    update = actual & np.any(delta != 0, axis=2)
    return dict(active=active, reference=reference, band=band, supported=supported,
                actual=actual, used=used, notes=notes, fields=fields, colour=colour,
                delta=delta, result=result, update=update)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_bundle(prior.OUT)
    rows = json.loads((prior.OUT / 'diagnostics.json').read_text())
    original_map = {r['page']['page']: r['page']['original'] for r in prior.diagnostic.previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    original_map.update({r['page']: r['original'] for r in json.loads((prior.diagnostic.fresh.INPUT / 'baseline_review.json').read_text())['cases']})
    deps = {str(Path(p).relative_to(ROOT)): sha(Path(p)) for p in [__file__, prior.__file__, prior.fill.__file__, metrics.__code__.co_filename]}
    for name in ['manifest.json', 'complete.json']:
        deps[str((prior.OUT / name).relative_to(ROOT))] = sha(prior.OUT / name)
    for path in original_map.values():
        deps[path] = sha(ROOT / path)
    manifest = dict(source_sha256=sha(Path(__file__)), dependencies=deps, settings=SETTINGS)
    if (OUT / 'manifest.json').exists():
        assert json.loads((OUT / 'manifest.json').read_text()) == manifest
    else:
        dump(OUT / 'manifest.json', manifest)
    shutil.copyfile(__file__, OUT / 'source_snapshot.py')
    results = []; poison = []; current = None
    for index, row in enumerate(rows, 1):
        rid = row['id']; key = row['page']; source = prior.OUT / 'native' / rid
        if key != current:
            original = read(ROOT / original_map[key])
            full_lab = cv2.cvtColor(original.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
            current = key; gc.collect()
        x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
        cs = (slice(cy, cy + ch), slice(cx, cx + cw))
        s = (slice(y - cy, y - cy + h), slice(x - cx, x - cx + w))
        lab = full_lab[cs]; crop = original[y:y+h, x:x+w]
        base = read(source / 'baseline.png'); app = read(source / 'applied_context.png', True)
        known = read(source / 'texture_reference_context.png', True)
        dest = OUT / 'native' / rid; dest.mkdir(parents=True, exist_ok=True)
        for name, a in dict(original=crop, baseline=base, applied_context=app, texture_reference_context=known).items():
            write(dest / (name + '.png'), a)
        outputs = {}; stats = {}; active_roi = bool(app[s].any())
        if active_roi:
            field = np.load(source / 'baseline_field_Lab.npy', allow_pickle=False)
            basecolour = np.load(source / 'baseline_color_Lab.npy', allow_pickle=False)
            assert np.array_equal(prior.fill.frozen.compose(field, app, base, s), base)
            np.save(dest / 'baseline_field_Lab.npy', field, allow_pickle=False)
            np.save(dest / 'baseline_color_Lab.npy', basecolour, allow_pickle=False)
        for radius in RADII:
            md = dest / f'radius{radius}'; md.mkdir(parents=True, exist_ok=True)
            active, reference, band = geometry(app, known, radius)
            r = dict(active=active, reference=reference, band=band, supported=np.zeros_like(app),
                     actual=np.zeros_like(app), used=np.zeros_like(app), update=np.zeros_like(app), notes=[])
            if active_roi:
                r = infer(lab, known, app, field, basecolour, radius)
                bad = lab.copy(); bad[~r['reference']] = (240, -210, 140)
                repeat = infer(bad, known, app, field, basecolour, radius)
                for name in ['active', 'reference', 'band', 'supported', 'actual', 'used', 'colour', 'delta', 'result', 'update']:
                    assert np.array_equal(r[name], repeat[name]), (rid, radius, name)
                assert r['notes'] == repeat['notes']
                for name in r['fields']:
                    assert np.array_equal(r['fields'][name], repeat['fields'][name])
                poison.append(dict(id=rid, radius=radius))
                np.savez_compressed(md / 'harmonic_fields_Lab.npz', **r['fields'])
                for name in ['colour', 'delta', 'result']:
                    np.save(md / (name + '_Lab.npy'), r[name], allow_pickle=False)
                value = prior.fill.frozen.compose(r['result'], r['update'], base, s)
                assert np.array_equal(r['result'][~r['actual']], field[~r['actual']])
            else:
                value = base.copy()
            assert not np.any(r['actual'] & ~app)
            assert np.array_equal(value[~r['actual'][s]], base[~r['actual'][s]])
            if radius == 0:
                assert np.array_equal(value, base), rid
            for name in ['active', 'reference', 'band', 'supported', 'actual', 'used', 'update']:
                write(md / (name + '_context.png'), r[name])
            write(md / 'fallback_context.png', app & ~r['actual'])
            write(md / 'output.png', value); dump(md / 'components.json', r['notes'])
            outputs[radius] = value
            stats[str(radius)] = {**metrics(value, base, app, known, s),
                'virtual_band_context': int(r['band'].sum()), 'reference_context': int(r['reference'].sum()),
                'supported_ROI': int(r['actual'][s].sum()), 'fallback_ROI': int((app & ~r['actual'])[s].sum())}
        figure([crop, base, outputs[3], outputs[6]],
               ['原画像', '現行補修・0px再計算一致', '紙色参照を3px離す', '紙色参照を6px離す'],
               OUT / 'comparisons' / (rid + '.png'), row['title'] + ' · 境界の紙色参照を離す',
               '仮未知領域は数値計算だけ。元の補間範囲内で紙色のみ更新、細かい元L模様と方向保留は固定。',
               'マスク拡張・模様補正・絶対パッチは併用なし。参照不足は現行補修を保持。')
        results.append({**row, 'original_source': original_map[key], 'setback_diagnostics': stats})
        dump(OUT / 'progress.json', dict(completed=index, total=90, latest=rid))
        print('Paper reference setback:', index, '/90', rid,
              [(radius, stats[str(radius)]['changed_vs_base'], stats[str(radius)]['fallback_ROI']) for radius in RADII], flush=True)
    for rel, digest in deps.items():
        assert sha(ROOT / rel) == digest, rel
    dump(OUT / 'diagnostics.json', results)
    dump(OUT / 'verification.json', dict(status='passed', pages=29, ROIs=90,
        radius0_all_outputs_exact_baseline=True, nonreference_and_virtual_band_poison=poison,
        outside_supported_applied_exact_baseline=True, no_new_applied_pixels=True,
        original_observed_texture_fixed=True, new_DDRM=False, pixel_GT=False))
    dump(OUT / 'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: paper reference setback, 90 ROIs', flush=True)


if __name__ == '__main__':
    main()
