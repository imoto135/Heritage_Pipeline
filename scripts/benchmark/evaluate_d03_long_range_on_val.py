"""Diagnostic transfer of the D03 exploratory long-range gate to existing val fade inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha, dump, rgb, mask, save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from evaluate_ridge_gate_independent_val import proxies
from probe_ridge_gate import features
from probe_d03_long_range_gate import outputs
from ragan_pilot_support import make_model

EXP = ROOT / 'tmp_work/makura_v5_exposure_loss_control_20261021'
BASE = EXP / 'independent_val_fade'
OUT = EXP / 'd03_long_range_probe/val_transfer'
METHODS = ('howe', 'ridge6_r3', 'q070_r12', 'q070_ridge6_r12')


def main():
    assert not (OUT / 'results.json').exists(), 'Preserve first transfer result'
    protocol = json.loads((BASE / 'protocol.json').read_text())
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    checkpoint = EXP / 'unet_v3_exposure/models/unet/last.pt'
    model = make_model('unet')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    model.to(device).eval()
    rows = []
    figures = []
    for patch in protocol['patches']:
        source = rgb(ROOT / 'tmp_work/makura_supervised_pilot_20261021/patches/val' / patch / 'clean.png')
        damaged_path = BASE / 'inputs' / (patch + '_alpha070.png')
        damaged = rgb(damaged_path)
        _, core, remote, distance = proxies(source)
        howe = mask(BASE / 'masks' / patch / 'howe.png')
        q = ink_probability(model, 'unet', damaged, device)
        _, ridge = features(damaged)
        preds = outputs(howe, q, ridge)
        items = [('原本', source), ('均一退色 α0.70', damaged)]
        for method in METHODS:
            pred = preds[method]
            path = OUT / 'masks' / patch / (method + '.png')
            save(path, pred)
            dilated = ndi.binary_dilation(pred, structure=np.array([[0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool))
            rows.append(dict(patch=patch, method=method, prediction_sha256=sha(path),
                             core_n=int(core.sum()), core_exact=int((pred & core).sum()),
                             core_r2=int((dilated & core).sum()),
                             remote_paper_n=int(remote.sum()), remote_paper_ink=int((pred & remote).sum()),
                             beyond_source_dark_2px=int((pred & (distance > 2)).sum()),
                             added_to_howe=int((pred & ~howe).sum())))
            items.append((method, np.where(pred, 0, 255).astype(np.uint8)))
        figure = OUT / 'figures' / (patch + '_原本比較.png')
        montage(figure, 'D03で選んだ遠距離規則の転移 / ' + patch, items, False)
        figures.append(dict(patch=patch, path=str(figure.relative_to(OUT)), sha256=sha(figure)))
    dump(OUT / 'results.json', dict(rows=rows, figures=figures, formal_accuracy=False,
        method_selection_after_d03_inspection=True, prior_val_images_already_inspected=True,
        checkpoint_sha256=sha(checkpoint)))
    for method in METHODS:
        a = [r for r in rows if r['method'] == method]
        print(method, 'core_r2', sum(r['core_r2'] for r in a), 'remote', sum(r['remote_paper_ink'] for r in a),
              'beyond2', sum(r['beyond_source_dark_2px'] for r in a), flush=True)


if __name__ == '__main__':
    main()
