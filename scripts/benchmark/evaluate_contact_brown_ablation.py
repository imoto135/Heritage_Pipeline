"""Read out same-field brown overlay vs multiplicative dark contact training."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from run_contact_stain_training_control import OUT
from evaluate_contact_stain_counterfactual import make_field
from evaluate_ridge_gate_independent_val import proxies
from compare_v5_aligned_fade_control import sha, dump, rgb, mask, save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

DEST = OUT / 'brown_aligned_evaluation'


def get_model(path, device):
    model = make_model('ragan')
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    return model.to(device).eval()


def main():
    assert not DEST.exists()
    DEST.mkdir()
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3, device)
    brown_path = OUT / 'ragan_brown_aligned/models/ragan/last.pt'
    dark_path = OUT / 'ragan_contact/models/ragan/last.pt'
    old_path = OUT / 'ragan_original/models/ragan/last.pt'
    models = {'old': get_model(old_path, device), 'brown_same_field': get_model(brown_path, device),
              'dark_same_field': get_model(dark_path, device)}
    patches = json.loads((OUT / 'protocol.json').read_text())['heldout_patches'] + \
              json.loads((OUT / 'howe_ragan_union_transfer/protocol.json').read_text())['patches']
    rows, figures = [], []
    footprint = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool)
    for patch in patches:
        first = patch in patches[:2]
        image_path = (OUT / 'evaluation/inputs' / f'{patch}_stain.png') if first else \
                     (OUT / 'howe_ragan_union_transfer/inputs' / f'{patch}.png')
        image = rgb(image_path)
        source = rgb(ROOT / 'tmp_work/makura_supervised_pilot_20261021/patches/val' / patch / 'clean.png')
        _, core, paper, _ = proxies(source)
        field, _ = make_field(core)
        core &= field >= .5
        paper &= field >= .5
        display = [('原本', source), ('暗い汚れ入力', image)]
        for name, model in models.items():
            pred = ink_probability(model, 'ragan', image, device) >= .5
            path = DEST / 'synthetic_masks' / patch / f'{name}.png'
            save(path, pred)
            rows.append(dict(patch=patch, cohort='first_two' if first else 'extra_six', method=name,
                source_core_n=int(core.sum()), source_core_r2=int((ndi.binary_dilation(pred, structure=footprint) & core).sum()),
                source_paper_n=int(paper.sum()), source_paper_ink=int((pred & paper).sum()),
                prediction_sha256=sha(path)))
            display.append((name, np.where(pred, 0, 255).astype(np.uint8)))
        if first:
            figure = DEST / 'figures' / f'{patch}_合成原本比較.png'
            montage(figure, '同じ汚れ配置：茶色半透明 vs 暗い乗算 / ' + patch, display)
            figures.append(dict(case=patch, path=str(figure.relative_to(DEST)), sha256=sha(figure)))
    real_cases = []
    classic = {c['id']: c for c in json.loads((ROOT / 'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']}
    for key in ('D03', 'D04'):
        c = classic[key]
        real_cases.append(dict(id=key, source=c['source'], source_sha256=c['source_sha256'], roi=c['roi']))
    real_cases.extend(json.loads((OUT / 'other_book_transfer/protocol.json').read_text())['cases'])
    real_rows = []
    for case in real_cases:
        key = case['id']
        source_path = ROOT / case['source']
        assert sha(source_path) == case['source_sha256']
        page = rgb(source_path)
        box = case['roi'] if 'roi' in case else case['crop_xywh']
        x, y, w, h = box
        raw = page[y:y+h, x:x+w]
        display = [('原本', raw)]
        for name, model in models.items():
            pred = real.infer_roi(model, 'ragan', page, box, device) < .5
            path = DEST / 'real_masks' / key / f'{name}.png'
            save(path, pred)
            display.append((name, np.where(pred, 0, 255).astype(np.uint8)))
            if key == 'D04':
                rx, ry, rw, rh = json.loads((OUT / 'd04_spot_control/protocol.json').read_text())['spot_box_xywh']
            elif key == 'T07':
                rx, ry, rw, rh = json.loads((OUT / 'other_book_transfer/protocol.json').read_text())['T07_stain_paper_box_xywh']
            elif key == 'C01':
                rx, ry, rw, rh = json.loads((OUT / 'other_book_transfer/protocol.json').read_text())['C01_spot_box_xywh']
            else:
                continue
            region = pred[ry:ry+rh, rx:rx+rw]
            real_rows.append(dict(case=key, method=name, selected_region_n=rw*rh,
                selected_region_ink=int(region.sum()), prediction_sha256=sha(path)))
        figure = DEST / 'figures' / f'{key}_実損傷原本比較.png'
        real.panel(figure, key + ' / 同じ汚れ配置の茶色合成 vs 暗い合成', display)
        figures.append(dict(case=key, path=str(figure.relative_to(DEST)), sha256=sha(figure)))
    dump(DEST / 'results.json', dict(synthetic=rows, real_selected_regions=real_rows, figures=figures,
        checkpoints={n: sha(p) for n, p in [('old', old_path), ('brown_same_field', brown_path), ('dark_same_field', dark_path)]},
        same_field_brown_vs_dark=True, two_val_groups_only=True, formal_accuracy=False))
    for cohort in ('first_two', 'extra_six'):
        for method in models:
            a = [r for r in rows if r['cohort'] == cohort and r['method'] == method]
            print(cohort, method, 'core', sum(r['source_core_r2'] for r in a), '/', sum(r['source_core_n'] for r in a),
                  'paper', sum(r['source_paper_ink'] for r in a), '/', sum(r['source_paper_n'] for r in a), flush=True)
    for r in real_rows:
        print('REAL', r['case'], r['method'], r['selected_region_ink'], '/', r['selected_region_n'], flush=True)


if __name__ == '__main__':
    main()
