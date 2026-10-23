"""Fixed ridge-gate transfer on uninspected validation groups with source-only proxies."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha, dump, rgb, save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from probe_ridge_gate import features, variants
from ragan_pilot_support import make_model

EXP = ROOT / 'tmp_work/makura_v5_exposure_loss_control_20261021'
OUT = EXP / 'independent_val_fade'


def close_contrast(source):
    paper = cv2.morphologyEx(source, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    gs = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.int16)
    gp = cv2.cvtColor(paper, cv2.COLOR_RGB2GRAY).astype(np.int16)
    return paper, gp - gs


def proxies(source):
    paper, contrast = close_contrast(source)
    components, labels, stats, _ = cv2.connectedComponentsWithStats((contrast >= 30).astype(np.uint8), 8)
    large = np.zeros(contrast.shape, bool)
    for i in range(1, components):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            large[labels == i] = True
    core = skeletonize(large)
    dark = contrast >= 20
    distance = ndi.distance_transform_edt(~dark)
    remote = (contrast <= 5) & (distance >= 8)
    return paper, core, remote, distance


def howe_mask(damaged, folder, tag, howe):
    g = howe.deep.base.gray(damaged)
    terms, _ = howe.terms(g, thi=.4)
    [(pre, _)] = list(howe.sweep(terms, [160], folder, tag))
    pred, _ = howe.post(pre)
    assert pred.shape == damaged.shape[:2]
    return pred


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert not (OUT / 'results.json').exists(), 'Preserve the first fixed-rule transfer result'
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    checkpoint = EXP / 'unet_v3_exposure/models/unet/last.pt'
    model = make_model('unet')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    model.to(device).eval()
    spec = importlib.util.spec_from_file_location('howe', ROOT / 'tmp_work/howe_diagnosis_20261008/experiment.py')
    howe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = howe
    spec.loader.exec_module(howe)
    rows = []
    inputs = []
    figures = []
    for patch in protocol['patches']:
        source_path = ROOT / 'tmp_work/makura_supervised_pilot_20261021/patches/val' / patch / 'clean.png'
        source = rgb(source_path)
        paper, core, remote, distance = proxies(source)
        assert core.any() and remote.any()
        damaged = np.rint(source.astype(np.float32) * .3 + paper.astype(np.float32) * .7).astype(np.uint8)
        input_path = OUT / 'inputs' / (patch + '_alpha070.png')
        input_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(damaged).save(input_path)
        deltas = cv2.cvtColor(damaged, cv2.COLOR_RGB2GRAY).astype(np.int16) - cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.int16)
        inputs.append(dict(patch=patch, source_sha256=sha(source_path), input_sha256=sha(input_path),
                           source_core_n=int(core.sum()), remote_paper_n=int(remote.sum()),
                           core_changed_ge10=int(((deltas >= 10) & core).sum()),
                           median_core_brightening=float(np.median(deltas[core]))))
        howe_pred = howe_mask(damaged, OUT / 'scratch' / patch, 'howe_fixed', howe)
        q = ink_probability(model, 'unet', damaged, device)
        local, ridge = features(damaged)
        outputs = variants(howe_pred, q, local, ridge)
        display = [('損傷前原本', source), ('均一退色 α0.70', damaged)]
        for method in protocol['methods']:
            pred = outputs[method]
            path = OUT / 'masks' / patch / (method + '.png')
            save(path, pred)
            dilated = ndi.binary_dilation(pred, structure=np.array([[0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool))
            rows.append(dict(patch=patch, method=method, prediction_sha256=sha(path),
                             core_n=int(core.sum()), core_exact=int((pred & core).sum()),
                             core_r2=int((dilated & core).sum()),
                             remote_paper_n=int(remote.sum()), remote_paper_ink=int((pred & remote).sum()),
                             beyond_source_dark_2px=int((pred & (distance > 2)).sum()),
                             total_ink=int(pred.sum()), added_to_howe=int((pred & ~howe_pred).sum())))
            display.append((method, np.where(pred, 0, 255).astype(np.uint8)))
        figure = OUT / 'figures' / (patch + '_原本比較.png')
        montage(figure, '未使用検証ページ / 固定規則 / ' + patch, display, False)
        figures.append(dict(patch=patch, path=str(figure.relative_to(OUT)), sha256=sha(figure)))
        print(patch, 'core', int(core.sum()), 'remote', int(remote.sum()),
              'Howe', rows[-5]['core_r2'], 'ridge6', rows[-3]['core_r2'], flush=True)
    dump(OUT / 'results.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
        checkpoint_sha256=sha(checkpoint), rows=rows, inputs=inputs, figures=figures,
        formal_accuracy=False, source_proxy_not_expert_truth=True))


if __name__ == '__main__':
    main()
