"""Separate rim and tone factors; v2 checks poison with each factor's own mask."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_paper_seam_texture as prior
from build_npj_v6_comparison import dump, read, sha, write
from package_paper_blending_review import boundary_metric, check_bundle

ROOT = prior.ROOT
OUT = ROOT / 'tmp_work/paper_rim_lowfrequency_pilot_v2_20261001'
MASKS = prior.prior.previous.mask_probe.OUT
MARGIN = prior.prior.previous.mask_probe.old
MODES = ['rim', 'tone24', 'tone48']
SETTINGS = dict(
    baseline='saved boundary + observed seam texture; all 22 development ROIs',
    mask_factor=dict(seed='frozen accepted dark-margin cores OR completed upper cores',
        halo_radius_px=6, min_original_L_drop=2,
        domain='saved safe domain excluding long lines, warm anchors, v6 red/extra',
        old_edit_union=True, reference='previous paper reference minus newly editable pixels',
        restoration='same harmonic and seam-texture algorithms; reference and donor eligibility track mask',
        full_page_mask=True, local_restoration=True, ROI_label_gating=False),
    tone_factor=dict(mask_applied_reference_texture_fixed=True, sigmas_px=[24, 48],
        guidance='normalized Gaussian of known paper only; subtract harmonic extension of guidance boundary',
        fade='zero at first inside pixel; full at distance >=9px',
        min_reference_mass=.05, delta_Lab_limits=[3, 1.5, 1.5],
        no_damage_blending=True),
    factors_combined=False, DDRM_recomputed=False, training=False, random_noise=False,
    pixel_GT=False, independent_accuracy=False, illustrations_excluded=True)


def rim_mask(old_edit, core, drop, domain, guard):
    halo = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    neighborhood = halo & domain & ~old_edit
    added = neighborhood & ~guard & (drop >= 2)
    return old_edit | added, added, dict(halo=halo, guarded=neighborhood & guard,
        below_drop=neighborhood & ~guard & (drop < 2))


def saved_base_field(native):
    raw = np.load(native / 'boundary_raw_field_Lab.npy', allow_pickle=False)
    smooth = np.load(native / 'boundary_smooth_field_Lab.npy', allow_pickle=False)
    applied = read(native / 'applied_context.png', True)
    distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.exp(-(np.maximum(distance-1, 0)/8.)**2) * applied
    field = smooth + (raw-smooth) * weight[:, :, None]
    seam = np.load(prior.OUT / 'native' / native.name / 'texture_seam.npy', allow_pickle=False)
    field[:, :, 0] += seam * np.clip((distance-1)/4., 0, 1) * applied
    return field, applied


def tone_guidance(lab, known, applied, sigma):
    mass = cv2.GaussianBlur(known.astype(np.float32), (0, 0), sigma)
    guide = cv2.GaussianBlur(np.where(known[:, :, None], lab, 0), (0, 0), sigma) / np.maximum(mass[:, :, None], 1e-6)
    fields, actual, used, notes = prior.prior.solve_fields(guide, known, applied)
    assert np.array_equal(actual, applied)
    delta = guide - fields['boundary_raw']
    distance = cv2.distanceTransform(applied.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((distance-1)/8., 0, 1) * applied * (mass >= .05)
    correction = np.clip(delta, -np.array([3, 1.5, 1.5], np.float32), np.array([3, 1.5, 1.5], np.float32)) * weight[:, :, None]
    return correction, guide, fields['boundary_raw'], used, mass, notes


def fill_rim(lab, known, active):
    fields, applied, used, harmonic_notes = prior.prior.solve_fields(lab, known, active)
    if applied.any():
        result, positions, notes, seam, average, sx, sy = prior.infer(lab, known, applied,
            fields['boundary_raw'], fields['boundary_smooth'])
        field = result['combined']
    else:
        field = np.zeros_like(lab); positions = []; notes = []
        seam = np.zeros(active.shape, np.float32); average = seam.copy()
        sx = np.full(active.shape, -1, np.int32); sy = sx.copy()
    return field, applied, used, harmonic_notes, positions, notes, seam, average, sx, sy


def compose(field, applied, base, crop_slice):
    value = base.copy()
    converted = prior.prior.previous.blending.previous.from_lab(field[crop_slice])
    value[applied[crop_slice]] = converted[applied[crop_slice]]
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in [prior.OUT, prior.prior.OUT, MASKS]: check_bundle(folder)
    check_bundle(MARGIN.OUT, 'candidate_manifest.json', 'candidate_complete.json')
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for module in [prior, prior.prior, MARGIN, MARGIN.first, MARGIN.second]:
        dependencies[str(Path(module.__file__).relative_to(ROOT))] = sha(Path(module.__file__))
    for folder in [prior.OUT, prior.prior.OUT, MASKS]:
        for name in ['manifest.json', 'complete.json']: dependencies[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    for name in ['candidate_manifest.json', 'candidate_complete.json']:
        dependencies[str((MARGIN.OUT/name).relative_to(ROOT))] = sha(MARGIN.OUT/name)
    manifest = dict(settings=SETTINGS, dependencies=dependencies, source_sha256=sha(Path(__file__)), pages=14, regions=22)
    if (OUT / 'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    prior_rows = {r['id']: r for r in json.loads((prior.prior.OUT/'diagnostics.json').read_text())}
    rows = []; page_rows = []; poison_cases = []
    for source in MARGIN.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; original = read(ROOT/page['original'])
            assert not page['contains_illustration']
            v6 = {k: read(source/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
            features, _ = MARGIN.load_features(key, original, v6)
            directory = MASKS/'pages'/key
            old_edit = read(directory/'completed/edit.png', True)
            domain = read(directory/'features/safe_domain.png', True)
            guard = read(directory/'features/warm_anchor_guard.png', True) | read(directory/'features/long_line_guard.png', True) | v6['red'] | v6['extra']
            core = read(MARGIN.OUT/'candidate_pages'/key/'features/accepted_core.png', True) | read(directory/'features/completed_core.png', True)
            edit, added, rejected = rim_mask(old_edit, core, features['light_drop'], domain, guard)
            assert not np.any(added & guard) and not np.any(added & ~domain)
            assert np.array_equal(edit & old_edit, old_edit)
            dest = OUT/'pages'/key
            for name, value in dict(old_edit=old_edit, rim_edit=edit, rim_added=added, core=core, guard=guard, domain=domain, **rejected).items():
                write(dest/(name+'.png'), value)
            page_rows.append(dict(page=key, added_pixels=int(added.sum()), guarded_pixels=int(rejected['guarded'].sum()),
                below_drop_pixels=int(rejected['below_drop'].sum())))
            for roi in page['rois']:
                rid = roi['id']; row = prior_rows[rid]
                cx, cy, cw, ch = row['context_xywh']; x, y, w, h = row['xywh']
                cs = (slice(cy, cy+ch), slice(cx, cx+cw))
                s = (slice(y-cy, y-cy+h), slice(x-cx, x-cx+w))
                context = original[cs]; lab = prior.prior.previous.blending.previous.to_lab(context)
                old_native = prior.prior.OUT/'native'/rid
                base = read(prior.OUT/'native'/rid/'combined.png')
                base_field, old_applied = saved_base_field(old_native)
                assert np.array_equal(compose(base_field, old_applied, base, s), base)
                known = read(old_native/'reference_context.png', True)
                old_eligible = read(old_native/'eligible_context.png', True)
                rim_known = known & ~edit[cs]
                rim_active = old_eligible | added[cs]
                local = OUT/'native'/rid
                local.mkdir(parents=True, exist_ok=True)
                if added[cs].any():
                    rfield, rapplied, used, hnotes, positions, qnotes, seam, average, sx, sy = fill_rim(lab, rim_known, rim_active)
                    rim = compose(rfield, rapplied, base, s)
                    for name, value in dict(rim_field_Lab=rfield, texture_seam=seam, texture_average=average, source_x=sx, source_y=sy).items():
                        np.save(local/(name+'.npy'), value, allow_pickle=False)
                    dump(local/'donors.json', dict(context_xywh=row['context_xywh'], patch_xy=positions, components=qnotes))
                    dump(local/'rim_components.json', hnotes)
                else:
                    rim = base.copy(); rapplied = old_applied.copy(); used = np.zeros_like(known)
                outputs = dict(rim=rim)
                corrected = {}
                for sigma in [24, 48]:
                    mode = 'tone'+str(sigma)
                    if old_applied.any():
                        correction, guide, hguide, gused, mass, notes = tone_guidance(lab, known, old_applied, sigma)
                    else:
                        correction = np.zeros_like(lab); guide = correction.copy(); hguide = correction.copy()
                        gused = np.zeros_like(known); mass = np.zeros_like(known, np.float32); notes = []
                    corrected[mode] = correction
                    outputs[mode] = compose(base_field+correction, old_applied, base, s)
                    for name, value in dict(correction_Lab=correction, guide_Lab=guide, harmonic_guide_Lab=hguide, reference_mass=mass).items():
                        np.save(local/(mode+'_'+name+'.npy'), value, allow_pickle=False)
                    write(local/(mode+'_used.png'), gused)
                if old_applied[s].any() or (added[cs]&rapplied)[s].any():
                    poisoned = context.copy(); poisoned[old_edit[cs]] = (241, 2, 13)
                    plab = prior.prior.previous.blending.previous.to_lab(poisoned)
                    for sigma in [24, 48]:
                        repeat = tone_guidance(plab, known, old_applied, sigma)[0]
                        assert np.array_equal(corrected['tone'+str(sigma)], repeat)
                    if added[cs].any():
                        poisoned = context.copy(); poisoned[edit[cs]] = (241, 2, 13)
                        plab = prior.prior.previous.blending.previous.to_lab(poisoned)
                        repeat = fill_rim(plab, rim_known, rim_active)
                        assert np.array_equal(repeat[0], rfield) and np.array_equal(repeat[1], rapplied)
                    poison_cases.append(rid)
                for name, value in dict(original=context[s], base=base, old_edit=old_edit[cs][s], rim_edit=edit[cs][s],
                    rim_added=added[cs][s], old_applied=old_applied[s], rim_applied=rapplied[s],
                    old_applied_context=old_applied, rim_applied_context=rapplied, reference_context=known,
                    rim_reference_context=rim_known, rim_eligible_context=rim_active, rim_used_context=used, **outputs).items():
                    write(local/(name+'.png'), value)
                diagnostics = {}
                # Use fixed old reference minus all proposed rim additions for fair boundary pairs.
                metric_known = rim_known[s]
                for mode, image in dict(base=base, **outputs).items():
                    allowed = rapplied[s] if mode == 'rim' else old_applied[s]
                    active_edit = edit[cs][s] if mode == 'rim' else old_edit[cs][s]
                    assert not np.any(np.any(image != base, axis=2) & ~allowed)
                    assert not np.any(np.any(image != context[s], axis=2) & ~active_edit)
                    count, median = boundary_metric(image, old_applied[s], metric_known)
                    L = prior.prior.previous.blending.previous.to_lab(image)[:, :, 0]
                    diagnostics[mode] = dict(changed_vs_base=int(np.any(image != base, axis=2).sum()),
                        fixed_old_boundary_pairs=count, fixed_old_boundary_median_Lab=median,
                        old_applied_L_le_80=int(((L <= 80)&old_applied[s]).sum()),
                        common_added_L_le_80=int(((L <= 80)&added[cs][s]).sum()))
                prior.prior.previous.prior_ddrm.figure([context[s], base, outputs['rim'], outputs['tone24'], outputs['tone48']],
                    ['原画像', '前回：境界＋紙模様', '輪郭回収のみ', '紙色のみ・σ24', '紙色のみ・σ48'],
                    OUT/'comparisons'/(rid+'.png'), row['title']+' · 輪郭と紙色を分離',
                    f'追加候補 {int(added[cs][s].sum()):,} / 追加補間 {int((added[cs]&rapplied)[s].sum()):,}画素。紙色列のマスクは固定。',
                    '輪郭列：同じ紙色補間・紙模様アルゴリズム。紙色列：固定参照・固定模様。DDRM再推論なし。')
                rows.append(dict(id=rid, page=key, title=row['title'], xywh=row['xywh'], context_xywh=row['context_xywh'],
                    rim_added_pixels=int(added[cs][s].sum()), rim_added_applied_pixels=int((added[cs]&rapplied)[s].sum()),
                    old_applied_pixels=int(old_applied[s].sum()), rim_applied_pixels=int(rapplied[s].sum()),
                    diagnostics=diagnostics, tone_correction_max={m: np.abs(v[s]).max(axis=(0, 1)).tolist() for m, v in corrected.items()}))
                print('Rim / tone:', len(rows), '/22', rid, 'added', int(added[cs][s].sum()), flush=True)
            del original, features, v6, old_edit, domain, guard, core, edit, added, rejected
            gc.collect()
    dump(OUT/'diagnostics.json', dict(pages=page_rows, regions=rows))
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'verification.json', dict(status='passed', pages=14, regions=22, factors_separate=True,
        original_damage_poison_cases=poison_cases, old_edit_preserved=True, guards_preserved=True,
        tone_mask_reference_texture_fixed=True, outside_each_edit_changed_pixels=0,
        DDRM_recomputed=False, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.json'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: separate rim-mask and fixed-mask paper guidance, 22 ROIs', flush=True)


if __name__ == '__main__': main()
