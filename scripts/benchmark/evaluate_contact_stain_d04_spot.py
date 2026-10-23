"""Known D04 isolated spot: check whether union reintroduces Howe's removed stain."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from run_contact_stain_training_control import OUT
from compare_v5_aligned_fade_control import sha, dump, rgb, mask, save
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

DEST = OUT / 'd04_spot_control'


def freeze():
    assert not DEST.exists()
    case = next(c for c in json.loads((ROOT / 'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases'] if c['id'] == 'D04')
    assert sha(ROOT / case['source']) == case['source_sha256']
    DEST.mkdir()
    dump(DEST / 'protocol.json', dict(case='D04', source=case['source'],
        source_sha256=case['source_sha256'], roi=case['roi'],
        spot_box_xywh=[120, 195, 85, 60], source_only_box=True,
        methods=['howe_fixed', 'sauvola_w61_k020', 'unet_contact_q50',
                 'ragan_contact_q50', 'howe_or_ragan_contact_q50'],
        selected_before_current_method_outputs=True, known_development_case=True,
        spot_is_provisional_nonink=True, expert_GT=False, formal_accuracy=False))


def main():
    protocol = json.loads((DEST / 'protocol.json').read_text())
    assert not (DEST / 'results.json').exists()
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3, device)
    models = {}
    for kind in ('unet', 'ragan'):
        ck = OUT / f'{kind}_contact/models/{kind}/last.pt'
        m = make_model(kind)
        m.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'])
        models[kind] = m.to(device).eval()
    source = ROOT / protocol['source']
    assert sha(source) == protocol['source_sha256']
    page = rgb(source)
    x, y, w, h = protocol['roi']
    crop = page[y:y+h, x:x+w]
    import cv2
    from evaluate_contact_stain_counterfactual import howe_mask
    spec = importlib.util.spec_from_file_location('howe_d04_contact', ROOT / 'tmp_work/howe_diagnosis_20261008/experiment.py')
    howe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = howe
    spec.loader.exec_module(howe)
    fixed = mask(ROOT / 'tmp_work/classical_limit_audit_20261021/crops/D04/howe_fixed.png')
    assert fixed.shape == (h, w)
    predictions = {'howe_fixed': fixed,
        'sauvola_w61_k020': howe.deep.base.sauvola(howe.deep.base.gray(crop), 61, .2),
        'unet_contact_q50': real.infer_roi(models['unet'], 'unet', page, protocol['roi'], device) < .5,
        'ragan_contact_q50': real.infer_roi(models['ragan'], 'ragan', page, protocol['roi'], device) < .5}
    predictions['howe_or_ragan_contact_q50'] = fixed | predictions['ragan_contact_q50']
    sx, sy, sw, sh = protocol['spot_box_xywh']
    region = np.zeros((h, w), bool)
    region[sy:sy+sh, sx:sx+sw] = True
    items = [('原画像', crop)]
    rows = []
    for method in protocol['methods']:
        pred = predictions[method]
        path = DEST / 'masks' / f'{method}.png'
        save(path, pred)
        rows.append(dict(method=method, spot_region_n=int(region.sum()), spot_region_ink=int((pred & region).sum()),
            prediction_sha256=sha(path)))
        items.append((method, np.where(pred, 0, 255).astype(np.uint8)))
    figure = DEST / 'D04_原本比較.png'
    real.panel(figure, 'D04 黒い斑点 / 接触汚れ学習の副作用', items)
    dump(DEST / 'results.json', dict(protocol_sha256=sha(DEST / 'protocol.json'),
        rows=rows, figure=str(figure.relative_to(DEST)), figure_sha256=sha(figure),
        known_development_case=True, provisional_spot_box=True, formal_accuracy=False))
    for row in rows:
        print(row['method'], row['spot_region_ink'], '/', row['spot_region_n'], flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'freeze':
        freeze()
    elif len(sys.argv) == 2 and sys.argv[1] == 'run':
        main()
    else:
        raise SystemExit('usage: ... freeze|run')
