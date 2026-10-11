"""Independently audit numerical reference setbacks and package PNG evidence."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import probe_paper_reference_setback as prior
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_raw_qualified_paper_colour_review import support_check
from package_paper_texture_blend_energy_review import check_metrics
from package_paper_blending_review import check_bundle

ROOT = prior.ROOT
OUT = ROOT / 'tmp_work/paper_reference_setback_review_20261001'


def synthetic_checks():
    app = np.zeros((48, 48), bool); app[17:31, 17:31] = True
    known = ~app
    colour = np.full((48, 48, 3), (90, 3, 12), np.float32)
    yy, xx = np.indices(app.shape)
    field = colour.copy(); field[:, :, 0] += .2 * np.sin(xx + yy)
    raw = colour.copy(); raw[app] = (0, 80, -80)
    for radius in prior.RADII:
        r = prior.infer(raw, known, app, field, colour, radius)
        assert np.array_equal(r['actual'], app)
        assert np.allclose(r['result'][app], field[app], atol=3e-5, rtol=0)
        poisoned = raw.copy(); poisoned[~r['reference']] = (200, -200, 200)
        repeat = prior.infer(poisoned, known, app, field, colour, radius)
        assert np.array_equal(r['result'], repeat['result'])
        unsupported = prior.infer(raw, np.zeros_like(app), app, field, colour, radius)
        assert not unsupported['actual'].any()
        assert np.array_equal(unsupported['result'], field)
    # A coloured ring excluded from the numerical boundary must not enter the solve.
    band = known & (distance_transform_edt(~app) <= 3)
    raw[band] = (70, 15, 25)
    immediate = prior.infer(raw, known, app, field, colour, 0)
    remote = prior.infer(raw, known, app, field, colour, 3)
    assert np.abs(immediate['colour'][app] - colour[app]).max() > 10
    assert np.allclose(remote['colour'][app], colour[app], atol=3e-5, rtol=0)
    assert np.array_equal(remote['result'][~app], field[~app])
    return dict(constant_paper_and_saved_texture_preserved=True,
                excluded_ring_cannot_supply_colour=True,
                no_reference_falls_back_to_baseline=True,
                image_update_remains_inside_original_applied=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_bundle(prior.OUT)
    synthetic = synthetic_checks()
    rows = json.loads((prior.OUT / 'diagnostics.json').read_text())
    checked = []; residuals = []; texture_errors = []
    unchanged = {str(r): [] for r in prior.RADII}
    current = None
    for row in rows:
        rid = row['id']; native = prior.OUT / 'native' / rid
        source = prior.prior.OUT / 'native' / rid
        if row['page'] != current:
            original = read(ROOT / row['original_source'])
            full_lab = cv2.cvtColor(original.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
            current = row['page']
        x, y, w, h = row['xywh']; cx, cy, cw, ch = row['context_xywh']
        cs = (slice(cy, cy + ch), slice(cx, cx + cw))
        s = (slice(y - cy, y - cy + h), slice(x - cx, x - cx + w))
        lab = full_lab[cs]
        app = read(native / 'applied_context.png', True)
        known = read(native / 'texture_reference_context.png', True)
        base = read(native / 'baseline.png')
        assert np.array_equal(read(native / 'original.png'), original[y:y+h, x:x+w])
        for name, value in [('applied_context', app), ('texture_reference_context', known), ('baseline', base)]:
            assert np.array_equal(value, read(source / (name + '.png'), value.dtype == bool))
        assert not np.any(app & known)
        # For these fixed integer radii, check the saved chamfer band against an
        # independent exact Euclidean transform as well as the prescribed transform.
        exact_distance = distance_transform_edt(~app) if app.any() else np.full(app.shape, np.inf)
        if app[s].any():
            field = np.load(native / 'baseline_field_Lab.npy', allow_pickle=False)
            colour = np.load(native / 'baseline_color_Lab.npy', allow_pickle=False)
            for name, value in [('baseline_field_Lab', field), ('baseline_color_Lab', colour)]:
                assert np.array_equal(value, np.load(source / (name + '.npy'), allow_pickle=False))
            distance = cv2.distanceTransform(app.astype(np.uint8), cv2.DIST_L2, 5)
            weight = np.exp(-(np.maximum(distance - 1, 0) / 8.) ** 2) * app
        for radius in prior.RADII:
            md = native / f'radius{radius}'
            masks = {n: read(md / (n + '_context.png'), True)
                     for n in ['active', 'reference', 'band', 'supported', 'actual', 'used', 'update', 'fallback']}
            band = known & (exact_distance <= radius)
            assert np.array_equal(masks['band'], band), (rid, radius, 'distance band')
            assert np.array_equal(masks['active'], app | band)
            assert np.array_equal(masks['reference'], known & ~band)
            assert np.array_equal(masks['actual'], app & masks['supported'])
            assert np.array_equal(masks['fallback'], app & ~masks['actual'])
            assert not np.any(masks['reference'] & masks['active'])
            assert not np.any(masks['update'] & ~app)
            value = read(md / 'output.png'); support = []
            if app[s].any():
                ref = masks['reference']; numerical = masks['supported']; actual = masks['actual']
                support = support_check(masks['active'], ref, numerical)
                notes = json.loads((md / 'components.json').read_text())
                assert len(notes) == len(support)
                for expected, note in zip(support, notes):
                    assert expected['component'] == note['id'] and expected['area'] == note['area']
                    assert expected['accepted'] == note['applied']
                    if 'boundary_pairs' in note:
                        assert expected['pairs'] == note['boundary_pairs']
                fields = np.load(md / 'harmonic_fields_Lab.npz', allow_pickle=False)
                mass = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 6)
                smooth = cv2.GaussianBlur(np.where(ref[:, :, None], lab, 0), (0, 0), 6) / np.maximum(mass[:, :, None], 1e-6)
                for mode, target in [('boundary_raw', lab), ('boundary_smooth', smooth)]:
                    error, used = verify_field(fields[mode], numerical, ref, target)
                    assert np.array_equal(used, masks['used'])
                    residuals.append(dict(id=rid, radius=radius, field=mode, maximum=error))
                newcolour = fields['boundary_smooth'] + (fields['boundary_raw'] - fields['boundary_smooth']) * weight[:, :, None]
                assert np.array_equal(newcolour, np.load(md / 'colour_Lab.npy', allow_pickle=False))
                delta = np.zeros_like(field); delta[actual] = newcolour[actual] - colour[actual]
                assert np.array_equal(delta, np.load(md / 'delta_Lab.npy', allow_pickle=False))
                updated = field + delta
                assert np.array_equal(updated, np.load(md / 'result_Lab.npy', allow_pickle=False))
                texture_error = float(np.abs((updated[actual] - newcolour[actual]) - (field[actual] - colour[actual])).max()) if actual.any() else 0.
                assert texture_error < 2e-5, (rid, radius, texture_error)
                texture_errors.append(dict(id=rid, radius=radius, max_float_error=texture_error))
                assert np.array_equal(updated[~actual], field[~actual])
                assert np.array_equal(masks['update'], actual & np.any(delta != 0, axis=2))
                converted = cv2.cvtColor(updated[s].astype(np.float32), cv2.COLOR_LAB2BGR)
                converted = np.rint(np.clip(converted, 0, 1) * 255).astype(np.uint8)
                expected = base.copy(); expected[masks['update'][s]] = converted[masks['update'][s]]
                assert np.array_equal(value, expected)
            else:
                for name in ['supported', 'actual', 'used', 'update']:
                    assert not masks[name].any()
                assert np.array_equal(value, base)
            assert np.array_equal(value[~masks['actual'][s]], base[~masks['actual'][s]])
            stat = row['setback_diagnostics'][str(radius)]
            check_metrics(value, base, app, known, s, stat)
            assert stat['virtual_band_context'] == int(band.sum())
            assert stat['reference_context'] == int(masks['reference'].sum())
            assert stat['supported_ROI'] == int(masks['actual'][s].sum())
            assert stat['fallback_ROI'] == int(masks['fallback'][s].sum())
            if stat['changed_vs_base'] == 0:
                unchanged[str(radius)].append(rid)
            if radius == 0:
                assert np.array_equal(value, base)
            checked.append(dict(id=rid, radius=radius, **stat, support=support))
        print('Audit paper reference setback:', len(checked) // 3, '/90', rid, flush=True)
    assert len(checked) == 270 and len(unchanged['0']) == 90
    assert len({r['page'] for r in rows}) == 29
    (OUT / 'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT / 'comparisons').glob('*.png'):
        shutil.copyfile(p, OUT / 'comparisons' / p.name)
    focus = ['unseen_thin_crack', 'extra_top_branch', 'extra_second_branch', 'hole_edge_new',
             'new_margin_hole', 'extra_brown_stain', 'new_brown_stamp', 'extra_isolated_marks', 'unseen_fold']
    for i, rid in enumerate(focus, 1):
        stack([OUT / 'comparisons' / (rid + '.png')], OUT / f'{i:02d}_{rid}.png',
              rid + ' · 境界から離した紙色参照',
              ['0px再計算は基準と完全一致。3／6pxの仮未知領域は数値用で実画像更新を広げない。',
               '元の細かい模様を固定。境界の色差増加と補修輪郭の反例を残す。'])
    dump(OUT / 'verification.json', dict(status='passed', pages=29, ROIs=90, conditions=270,
        records=checked, residuals=residuals, texture_errors=texture_errors, synthetic=synthetic,
        unchanged_ROIs=unchanged, radius0_all_outputs_exact_baseline=True,
        exact_Euclidean_band_checked=True, independent_component_boundary_support=True,
        independent_direct_harmonic_residual=True, saved_texture_retained_with_float_tolerance=True,
        update_subset_original_applied=True, fallback_and_outside_original_applied_exact=True,
        independent_output_metrics=True, new_DDRM=False, pixel_GT=False))
    notes = ROOT / 'tmp_work/paper_reference_setback_review_notes_20261001.md'
    (OUT / 'report.md').write_text(notes.read_text())
    pngs = sorted(OUT.rglob('*.png')); assert len(pngs) == 99
    with zipfile.ZipFile(OUT / 'comparison_pngs.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for p in pngs:
            archive.write(p, str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT / 'comparison_pngs.zip') as archive:
        assert len(archive.namelist()) == 99 and archive.testzip() is None
        for p in pngs:
            assert archive.read(str(p.relative_to(OUT))) == p.read_bytes()
    deps = {str(notes.relative_to(ROOT)): sha(notes)}
    for n in ['manifest.json', 'complete.json']:
        deps[str((prior.OUT / n).relative_to(ROOT))] = sha(prior.OUT / n)
    dump(OUT / 'delivery.json', dict(source_sha256=sha(Path(__file__)), PNG_count=99,
        dependencies=deps, outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.name != 'delivery.json'}))
    print('Delivered paper reference setback: 99 PNGs /270 conditions', flush=True)


if __name__ == '__main__':
    main()
