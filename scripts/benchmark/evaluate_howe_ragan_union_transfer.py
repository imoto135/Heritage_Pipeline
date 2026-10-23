"""Posthoc Howe/RA contact union, checked on six further same-group patches."""
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
from run_contact_stain_training_control import OUT, stain
from evaluate_contact_stain_counterfactual import make_field, howe_mask
from evaluate_ridge_gate_independent_val import proxies
from compare_v5_aligned_fade_control import sha, dump, rgb, save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from ragan_pilot_support import make_model

PATCHES = [f'200017197_{page}_p{index:02d}' for page in ('00171_1', '00283_0') for index in (1, 2, 3)]
DEST = OUT / 'howe_ragan_union_transfer'


def freeze():
    assert not DEST.exists()
    DEST.mkdir()
    dump(DEST / 'protocol.json', dict(patches=PATCHES,
        selection='all p01-p03 from the same two validation page groups; p00 was previously inspected',
        no_additional_parameter_search=True,
        damage='same center-nearest source proxy, multiplicative dark stain strength .65',
        methods=['howe_fixed', 'ragan_contact_q50', 'howe_or_ragan_contact_q50'],
        union_posthoc=True, page_groups_not_independent_of_first_two=True,
        threshold=.5, source_proxy_not_gt=True, formal_accuracy=False))


def main():
    protocol = json.loads((DEST / 'protocol.json').read_text())
    assert not (DEST / 'results.json').exists()
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3, device)
    checkpoint = OUT / 'ragan_contact/models/ragan/last.pt'
    model = make_model('ragan')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    model = model.to(device).eval()
    spec = importlib.util.spec_from_file_location('howe_union_transfer', ROOT / 'tmp_work/howe_diagnosis_20261008/experiment.py')
    howe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = howe
    spec.loader.exec_module(howe)
    rows, inputs, figures = [], [], []
    footprint = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool)
    for patch in protocol['patches']:
        source_path = ROOT / 'tmp_work/makura_supervised_pilot_20261021/patches/val' / patch / 'clean.png'
        source = rgb(source_path)
        paper, core, remote, _ = proxies(source)
        field, anchor = make_field(core)
        affected_core = core & (field >= .5)
        affected_paper = remote & (field >= .5)
        assert affected_core.any() and affected_paper.any()
        image = stain(source, field, .65)
        path = DEST / 'inputs' / f'{patch}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(path)
        inputs.append(dict(patch=patch, source_sha256=sha(source_path), input_sha256=sha(path),
            anchor_xy=anchor, source_core_n=int(affected_core.sum()), source_remote_paper_n=int(affected_paper.sum())))
        h = howe_mask(image, DEST / 'scratch' / patch, 'howe', howe)
        r = ink_probability(model, 'ragan', image, device) >= .5
        predictions = {'howe_fixed': h, 'ragan_contact_q50': r, 'howe_or_ragan_contact_q50': h | r}
        comparisons = [('原本', source), ('汚れ入力', image)]
        for method, pred in predictions.items():
            dest = DEST / 'masks' / patch / f'{method}.png'
            save(dest, pred)
            rows.append(dict(patch=patch, method=method, source_core_n=int(affected_core.sum()),
                source_core_exact=int((pred & affected_core).sum()),
                source_core_r2=int((ndi.binary_dilation(pred, structure=footprint) & affected_core).sum()),
                source_remote_paper_n=int(affected_paper.sum()),
                source_remote_paper_ink=int((pred & affected_paper).sum()),
                prediction_sha256=sha(dest)))
            comparisons.append((method, np.where(pred, 0, 255).astype(np.uint8)))
        fig = DEST / 'figures' / f'{patch}_原本比較.png'
        montage(fig, 'Howe + RA-GANの和集合 / 同一群の追加パッチ / ' + patch, comparisons)
        figures.append(dict(patch=patch, path=str(fig.relative_to(DEST)), sha256=sha(fig)))
        print(patch, 'core', int(affected_core.sum()), 'paper', int(affected_paper.sum()), flush=True)
    dump(DEST / 'results.json', dict(protocol_sha256=sha(DEST / 'protocol.json'),
        checkpoint_sha256=sha(checkpoint), rows=rows, inputs=inputs, figures=figures,
        union_selected_posthoc=True, same_page_groups=True, source_proxy_not_gt=True))
    for method in protocol['methods']:
        a = [r for r in rows if r['method'] == method]
        print(method, 'core_r2', sum(r['source_core_r2'] for r in a), '/', sum(r['source_core_n'] for r in a),
            'stain_paper_ink', sum(r['source_remote_paper_ink'] for r in a), '/',
            sum(r['source_remote_paper_n'] for r in a), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'freeze':
        freeze()
    elif len(sys.argv) == 2 and sys.argv[1] == 'run':
        main()
    else:
        raise SystemExit('usage: ... freeze|run')
