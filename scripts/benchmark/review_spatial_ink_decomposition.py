"""Read-only analysis of the frozen spatial pilot; no accuracy claims."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'tmp_work/ink_background_spatial_pilot_v3_20260930'
OUT = ROOT / 'tmp_work/ink_background_spatial_review_20260930'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    frozen = json.loads((SOURCE / 'complete.json').read_text())
    for name, expected in frozen['outputs'].items():
        assert sha(SOURCE / name) == expected, name
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for name, expected in manifest['dependencies'].items():
        assert sha(ROOT / name) == expected, name
    assert sha(ROOT / 'scripts/benchmark/probe_spatial_ink_decomposition.py') == manifest['source_sha256']
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in json.loads((SOURCE / 'diagnostics.json').read_text()):
        page, roi = entry['page'], entry['roi']
        x, y, w, h = entry['xywh']
        folder = SOURCE / page / roi
        densities = {}
        for name in entry['fits']:
            a = np.load(folder / (name + '.npz'))['density']
            assert np.isfinite(a).all() and (a >= 0).all()
            densities[name] = cv2.resize(a, (w + 96, h + 96), interpolation=cv2.INTER_LINEAR)[48:48+h, 48:48+w]
        ablations = {}
        for bg in ['smooth', 'edge_aware']:
            g, n = densities[bg + '_geometry_q1'], densities[bg + '_no_geometry_q1']
            gm, nm = g >= .015, n >= .015
            ablations[bg] = dict(relative_l1_change=float(np.abs(g-n).sum()/max(n.sum(), 1e-12)),
                geometry_above_gate=int(gm.sum()), no_geometry_above_gate=int(nm.sum()),
                mask_iou=float((gm & nm).sum()/max((gm | nm).sum(), 1)),
                added_pixels=int((gm & ~nm).sum()), removed_pixels=int((nm & ~gm).sum()),
                display_clipped_fraction=float((g >= .2).mean()))
        saved = np.load(folder / 'native_diagnostics.npz')
        family = np.stack([v for k, v in densities.items() if '_geometry_' in k])
        assert np.array_equal(saved['minimum_density'], family.min(0))
        assert np.array_equal(saved['maximum_density'], family.max(0))
        rows.append(dict(page=page, roi=roi, xywh=entry['xywh'], ablations=ablations,
            v6_editable_pixels=entry['v6_editable_pixels'],
            stable_on_v6_edit_pixels=entry['stable_on_v6_edit_pixels'],
            all_models_above_gate_pixels=entry['all_models_above_gate_pixels']))
    summary = dict(rows=rows, diagnostic_threshold=.015,
        statistics='Bilinearly upsampled uncolored density on original ROI; not independent pixels or GT',
        geometry_ablation='Tangent regularization and ridge-based sparsity changed together; not orientation alone',
        improvement_over_v6_demonstrated=False, pixel_GT=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    verification = dict(status='passed', frozen_output_hashes_checked=len(frozen['outputs']),
        dependency_hashes_checked=len(manifest['dependencies']), density_range_reproduced=True,
        original_pilot_modified=False, numerical_solver_checks=json.loads((SOURCE/'numerical_checks.json').read_text()),
        browser_rendering_tested=False, accuracy_evaluated=False)
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()
