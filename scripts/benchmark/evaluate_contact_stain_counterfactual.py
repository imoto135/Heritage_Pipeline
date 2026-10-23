"""Force a procedural stain across source-dark strokes on untouched val groups."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha, dump, rgb, save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from evaluate_ridge_gate_independent_val import proxies
from probe_ridge_gate import features, variants
from probe_d03_long_range_gate import outputs as long_outputs
from ragan_pilot_support import make_model

EXP = ROOT / 'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT = EXP / 'contact_stain_counterfactual'


def make_field(core):
    yx = np.argwhere(core)
    inside = (yx[:, 0] >= 128) & (yx[:, 0] < 384) & (yx[:, 1] >= 128) & (yx[:, 1] < 384)
    yx = yx[inside]
    assert len(yx) > 0
    cy, cx = yx[np.argmin(np.sum((yx - 256) ** 2, axis=1))]
    yy, xx = np.mgrid[:core.shape[0], :core.shape[1]]
    dx, dy = xx - cx, yy - cy
    angle = .4
    u = (dx * np.cos(angle) + dy * np.sin(angle)) / 110
    v = (-dx * np.sin(angle) + dy * np.cos(angle)) / 75
    theta = np.arctan2(v, u)
    radius = np.sqrt(u * u + v * v) / (1 + .13 * np.sin(3 * theta + 1.1) + .08 * np.sin(7 * theta + 2.2))
    field = np.clip((1 - radius) / .32 + .5, 0, 1).astype(np.float32)
    field[field < .01] = 0
    return field, (int(cx), int(cy))


def stain(source, field):
    f = field * .65
    return np.rint(np.clip(source.astype(np.float32) * np.exp(-f[..., None] *
        np.array([2., 2.6, 3.3], np.float32)), 0, 255)).astype(np.uint8)


def howe_mask(damaged, folder, tag, howe):
    gray = howe.deep.base.gray(damaged)
    terms, _ = howe.terms(gray, thi=.4)
    [(pre, _)] = list(howe.sweep(terms, [160], folder, tag))
    pred, _ = howe.post(pre)
    assert pred.shape == damaged.shape[:2]
    return pred


def load_model(kind, checkpoint, device):
    model = make_model(kind)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    return model.to(device).eval()


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert not (OUT / 'results.json').exists(), 'Preserve first counterfactual'
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    unet_ck = EXP / 'unet_v3_exposure/models/unet/last.pt'
    ragan_ck = ROOT / 'tmp_work/makura_v5_aligned_fade_control_20261021/ragan_public_v3_v5/models/ragan/last.pt'
    unet = load_model('unet', unet_ck, device)
    ragan = load_model('ragan', ragan_ck, device)
    spec = importlib.util.spec_from_file_location('howe', ROOT / 'tmp_work/howe_diagnosis_20261008/experiment.py')
    howe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = howe
    spec.loader.exec_module(howe)
    rows, inputs, figures = [], [], []
    structure = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool)
    for patch in protocol['patches']:
        source_path = ROOT / 'tmp_work/makura_supervised_pilot_20261021/patches/val' / patch / 'clean.png'
        source = rgb(source_path)
        paper, core, remote, distance = proxies(source)
        field, anchor = make_field(core)
        overlap = field >= .5
        affected_core = core & overlap
        stain_paper = remote & overlap
        assert int(affected_core.sum()) >= 100 and int(stain_paper.sum()) >= 1000
        damaged = stain(source, field)
        counter = stain(paper, field)
        signal = np.max(np.abs(damaged.astype(np.int16) - counter.astype(np.int16)), axis=2)
        inp = OUT / 'inputs' / (patch + '_contact_stain.png')
        inp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(damaged).save(inp)
        Image.fromarray(np.rint(field * 255).astype(np.uint8)).save(OUT / 'inputs' / (patch + '_field.png'))
        inputs.append(dict(patch=patch, source_sha256=sha(source_path), input_sha256=sha(inp),
            anchor_xy=anchor, affected_core_n=int(affected_core.sum()), stain_paper_n=int(stain_paper.sum()),
            affected_core_median_remaining_rgb_contrast=float(np.median(signal[affected_core])),
            affected_core_signal_lt8=int((signal[affected_core] < 8).sum())))
        howe_pred = howe_mask(damaged, OUT / 'scratch' / patch, 'howe_fixed', howe)
        g = howe.deep.base.gray(damaged)
        sauv = howe.deep.base.sauvola(g, 61, .2)
        uq = ink_probability(unet, 'unet', damaged, device)
        rq = ink_probability(ragan, 'ragan', damaged, device)
        local, ridge = features(damaged)
        predictions = {
            'howe_fixed': howe_pred,
            'sauvola_w61_k020': sauv,
            'unet_q020': uq >= .2,
            'howe_ridge6_r3': variants(howe_pred, uq, local, ridge)['ridge6_r3'],
            'howe_q070_r12': long_outputs(howe_pred, uq, ridge)['q070_r12'],
            'ragan_public_q050': rq >= .5,
        }
        display = [('損傷前原本', source), ('文字に接する合成汚れ', damaged)]
        for method in protocol['methods']:
            pred = predictions[method]
            path = OUT / 'masks' / patch / (method + '.png')
            save(path, pred)
            dilated = ndi.binary_dilation(pred, structure=structure)
            rows.append(dict(patch=patch, method=method, prediction_sha256=sha(path),
                affected_core_n=int(affected_core.sum()), affected_core_exact=int((pred & affected_core).sum()),
                affected_core_r2=int((dilated & affected_core).sum()),
                stain_paper_n=int(stain_paper.sum()), stain_paper_ink=int((pred & stain_paper).sum()),
                overlap_ink=int((pred & overlap).sum()), added_to_howe=int((pred & ~howe_pred).sum())))
            display.append((method, np.where(pred, 0, 255).astype(np.uint8)))
        figure = OUT / 'figures' / (patch + '_原本比較.png')
        montage(figure, '文字に接する合成汚れ / ' + patch, display, False)
        figures.append(dict(patch=patch, path=str(figure.relative_to(OUT)), sha256=sha(figure)))
        print(patch, 'affected core', int(affected_core.sum()), 'stain paper', int(stain_paper.sum()),
              'median signal', float(np.median(signal[affected_core])), flush=True)
    dump(OUT / 'results.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
        unet_checkpoint_sha256=sha(unet_ck), ragan_checkpoint_sha256=sha(ragan_ck),
        inputs=inputs, rows=rows, figures=figures, formal_accuracy=False, source_proxy_not_gt=True))
    for method in protocol['methods']:
        r = [x for x in rows if x['method'] == method]
        print(method, 'core_r2', sum(x['affected_core_r2'] for x in r),
              'stain_paper_ink', sum(x['stain_paper_ink'] for x in r), flush=True)


if __name__ == '__main__':
    main()
