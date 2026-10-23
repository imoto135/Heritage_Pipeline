"""Apply the fixed development ridge gate to other classical-limit crops."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha, dump, rgb, mask, save
from probe_ridge_gate import features, variants
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

EXP = ROOT / 'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT = EXP / 'ridge_gate_transfer'
CASE_IDS = ('D01', 'D02', 'D03', 'D04', 'T01', 'L07')
METHODS = ('howe', 'howe_plus_r3', 'ridge6_r3', 'ridge10_r3',
           'contrast16_ridge6_r3', 'contrast16_ridge6_r8')


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    checkpoint = EXP / 'unet_v3_exposure/models/unet/last.pt'
    model = make_model('unet')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    model.to(device).eval()

    spec = importlib.util.spec_from_file_location('source_scoring', ROOT / 'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scoring
    spec.loader.exec_module(scoring)
    part = ROOT / 'tmp_work/classical_partial_labels_20261021'
    regions = json.loads((part / 'source_regions.json').read_text())
    annotations = {r['id']: r for r in json.loads((part / 'annotations_source_only.json').read_text())['regions']}
    cases = json.loads((ROOT / 'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    rows = []
    figures = []
    for case in (c for c in cases if c['id'] in CASE_IDS):
        page = rgb(ROOT / case['source'])
        x, y, w, h = case['roi']
        raw = page[y:y+h, x:x+w]
        howe = mask(ROOT / 'tmp_work/classical_limit_audit_20261021/crops' / case['id'] / 'howe_fixed.png')
        q = 1 - real.infer_roi(model, 'unet', page, case['roi'], device)
        local, ridge = features(page)
        outputs = variants(howe, q, local[y:y+h, x:x+w], ridge[y:y+h, x:x+w])
        items = [('原画像', raw)]
        for name in METHODS:
            pred = outputs[name]
            dest = OUT / 'masks' / case['id'] / (name + '.png')
            save(dest, pred)
            items.append((name, np.where(pred, 0, 255).astype(np.uint8)))
            base = dict(case=case['id'], method=name, ink_pixels=int(pred.sum()),
                        added_to_howe=int((pred & ~howe).sum()), prediction_sha256=sha(dest))
            found = False
            for region in regions:
                if region['case'] != case['id']:
                    continue
                found = True
                labels, paths = scoring.raster(region, annotations[region['id']])
                rx, ry, rw, rh = region['roi']
                measurements = scoring.measure(pred[ry-y:ry-y+rh, rx-x:rx-x+rw], labels, paths)
                rows.append(dict(**base, region=region['id'], **measurements))
            if not found:
                rows.append(dict(**base, region=None))
        dest = OUT / 'figures' / (case['id'] + '_原画像比較.png')
        real.panel(dest, case['id'] + ' / 固定した線形状規則の転移', items)
        figures.append(dict(case=case['id'], path=str(dest.relative_to(OUT)), sha256=sha(dest)))
        print(case['id'], 'Howe', int(howe.sum()), 'ridge6 added', int((outputs['ridge6_r3'] & ~howe).sum()), flush=True)
    dump(OUT / 'results.json', dict(rows=rows, figures=figures, checkpoint_sha256=sha(checkpoint),
        case_ids=CASE_IDS, method_ids=METHODS, parent_rule='probe_ridge_gate.py fixed before this transfer',
        development_not_blind=True, formal_accuracy=False, partial_labels_source_only=True))


if __name__ == '__main__':
    main()
