"""Exploratory long-range Howe-U-Net gates on D03 and D04 known cases."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha, dump, rgb, mask, save
from probe_ridge_gate import features
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

EXP = ROOT / 'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT = EXP / 'd03_long_range_probe'


def outputs(howe, q, ridge):
    dist = ndi.distance_transform_edt(~howe)
    return {'howe': howe,
            'ridge6_r3': howe | ((q >= .2) & (dist <= 3) & (ridge >= 6)),
            'q070_r12': howe | ((q >= .7) & (dist <= 12)),
            'q070_ridge6_r12': howe | ((q >= .7) & (dist <= 12) & (ridge >= 6)),
            'q080_ridge6_r12': howe | ((q >= .8) & (dist <= 12) & (ridge >= 6)),
            'q070_ridge6_r16': howe | ((q >= .7) & (dist <= 16) & (ridge >= 6))}


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert not (OUT / 'results.json').exists(), 'Preserve first exploratory result'
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    checkpoint = EXP / 'unet_v3_exposure/models/unet/last.pt'
    model = make_model('unet')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    model.to(device).eval()
    spec = importlib.util.spec_from_file_location('scoring', ROOT / 'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scoring
    spec.loader.exec_module(scoring)
    part = ROOT / 'tmp_work/classical_partial_labels_20261021'
    regions = json.loads((part / 'source_regions.json').read_text())
    anns = {r['id']: r for r in json.loads((part / 'annotations_source_only.json').read_text())['regions']}
    cases = json.loads((ROOT / 'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
    rows = []
    figures = []
    for case in (c for c in cases if c['id'] in protocol['cases']):
        page = rgb(ROOT / case['source'])
        x, y, w, h = case['roi']
        raw = page[y:y+h, x:x+w]
        howe = mask(ROOT / 'tmp_work/classical_limit_audit_20261021/crops' / case['id'] / 'howe_fixed.png')
        q = 1 - real.infer_roi(model, 'unet', page, case['roi'], device)
        _, ridge_full = features(page)
        ridge = ridge_full[y:y+h, x:x+w]
        preds = outputs(howe, q, ridge)
        items = [('原画像', raw)]
        for method in protocol['methods']:
            pred = preds[method]
            path = OUT / 'masks' / case['id'] / (method + '.png')
            save(path, pred)
            items.append((method, np.where(pred, 0, 255).astype(np.uint8)))
            base = dict(case=case['id'], method=method, prediction_sha256=sha(path),
                        added_to_howe=int((pred & ~howe).sum()))
            if case['id'] == 'D03':
                for region in regions:
                    if region['case'] != 'D03':
                        continue
                    labels, paths = scoring.raster(region, anns[region['id']])
                    rx, ry, rw, rh = region['roi']
                    score = scoring.measure(pred[ry-y:ry-y+rh, rx-x:rx-x+rw], labels, paths)
                    rows.append(dict(**base, region=region['id'], **score))
            else:
                # Previously designated spot rectangle, broad and not a dense GT.
                rect = case['probes'][0]['negatives'][0]
                rx, ry, rw, rh = rect
                rows.append(dict(**base, region='spot_rectangle', spot_pixels=int(pred[ry-y:ry-y+rh, rx-x:rx-x+rw].sum()),
                                 rectangle=rect))
        figure = OUT / 'figures' / (case['id'] + '_原画像比較.png')
        real.panel(figure, case['id'] + ' / Howeから離れた候補の探索', items)
        figures.append(dict(case=case['id'], path=str(figure.relative_to(OUT)), sha256=sha(figure)))
    dump(OUT / 'results.json', dict(rows=rows, figures=figures, checkpoint_sha256=sha(checkpoint),
         protocol_sha256=sha(OUT / 'protocol.json'), formal_accuracy=False))
    for row in rows:
        if row['region'] == 'contact_upper':
            print('D03', row['method'], 'upper', next(p for p in row['paths'] if p['id']=='upper_stroke')['hit_r2'],
                  'nonink', row['nonink_false'], flush=True)
        if row['region'] == 'spot_rectangle':
            print('D04', row['method'], 'spot_rectangle', row['spot_pixels'], flush=True)


if __name__ == '__main__':
    main()
