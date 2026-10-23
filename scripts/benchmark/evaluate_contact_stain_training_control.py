"""Frozen heldout page-group evaluation of contact-stain training controls."""
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
from run_contact_stain_training_control import OUT, stain
from evaluate_contact_stain_counterfactual import make_field, howe_mask
from evaluate_ridge_gate_independent_val import proxies
from compare_v5_aligned_fade_control import sha, dump, rgb, save
from audit_v5_probability_thresholds import ink_probability
from audit_targeted_thin_fade import montage
from ragan_pilot_support import make_model


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    evaluation = OUT / 'evaluation'
    assert not (evaluation / 'results.json').exists(), 'Keep first heldout readout immutable'
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3, device)
    models = {}
    for kind in ('unet', 'ragan'):
        for variant in ('original', 'contact'):
            checkpoint = OUT / f'{kind}_{variant}' / 'models' / kind / 'last.pt'
            assert checkpoint.exists(), checkpoint
            model = make_model(kind)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
            models[f'{kind}_{variant}'] = (model.to(device).eval(), kind, sha(checkpoint))
    spec = importlib.util.spec_from_file_location('howe_contact_training', ROOT / 'tmp_work/howe_diagnosis_20261008/experiment.py')
    howe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = howe
    spec.loader.exec_module(howe)
    rows, inputs, figures = [], [], []
    footprint = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool)
    for patch in protocol['heldout_patches']:
        source_path = ROOT / 'tmp_work/makura_supervised_pilot_20261021/patches/val' / patch / 'clean.png'
        source = rgb(source_path)
        paper, core, remote, _ = proxies(source)
        field, anchor = make_field(core)
        overlap = field >= .5
        affected_core = core & overlap
        affected_paper = remote & overlap
        assert affected_core.sum() >= 100 and affected_paper.sum() >= 1000
        damaged = stain(source, field, protocol['eval_strength'])
        clean_counter = stain(paper, field, protocol['eval_strength'])
        remaining = np.max(np.abs(damaged.astype(np.int16) - clean_counter.astype(np.int16)), axis=2)
        input_path = evaluation / 'inputs' / f'{patch}_stain.png'
        input_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(damaged).save(input_path)
        Image.fromarray(np.rint(field * 255).astype(np.uint8)).save(evaluation / 'inputs' / f'{patch}_field.png')
        inputs.append(dict(patch=patch, source_sha256=sha(source_path), input_sha256=sha(input_path),
            anchor_xy=anchor, affected_source_core=int(affected_core.sum()),
            affected_remote_paper=int(affected_paper.sum()),
            median_remaining_rgb_contrast=float(np.median(remaining[affected_core])),
            low_signal_core=int((remaining[affected_core] < 8).sum())))
        for case, image in [('clean', source), ('contact_stain', damaged)]:
            comparisons = [('原本', source), ('評価入力', image)]
            predictions = {}
            predictions['howe_fixed'] = howe_mask(image, evaluation / 'scratch' / patch / case, 'howe', howe)
            g = howe.deep.base.gray(image)
            predictions['sauvola_w61_k020'] = howe.deep.base.sauvola(g, 61, .2)
            for name, (model, kind, _) in models.items():
                q = ink_probability(model, kind, image, device)
                for threshold in protocol['eval_thresholds'][kind]:
                    predictions[f'{name}_q{int(threshold*100):02d}'] = q >= threshold
            for method, pred in predictions.items():
                destination = evaluation / 'masks' / patch / case / f'{method}.png'
                save(destination, pred)
                measured_core = affected_core if case == 'contact_stain' else core
                measured_paper = affected_paper if case == 'contact_stain' else remote
                dilated = ndi.binary_dilation(pred, structure=footprint)
                rows.append(dict(patch=patch, case=case, method=method,
                    checkpoint_sha256=models[method.rsplit('_q', 1)[0]][2] if '_q' in method else None,
                    prediction_sha256=sha(destination), source_core_n=int(measured_core.sum()),
                    source_core_exact=int((pred & measured_core).sum()),
                    source_core_r2=int((dilated & measured_core).sum()),
                    source_remote_paper_n=int(measured_paper.sum()),
                    source_remote_paper_ink=int((pred & measured_paper).sum()),
                    total_ink=int(pred.sum())))
                comparisons.append((method, np.where(pred, 0, 255).astype(np.uint8)))
            figure = evaluation / 'figures' / f'{patch}_{case}_原本比較.png'
            montage(figure, '接触汚れ学習 / 未学習ページ群 / ' + patch + ' / ' + case, comparisons)
            figures.append(dict(patch=patch, case=case, path=str(figure.relative_to(evaluation)), sha256=sha(figure)))
        print(patch, 'contact core', int(affected_core.sum()), 'stain paper', int(affected_paper.sum()), flush=True)
    dump(evaluation / 'results.json', dict(protocol_sha256=sha(OUT / 'protocol.json'),
        inputs=inputs, rows=rows, figures=figures, source_proxy_not_gt=True, formal_accuracy=False))
    for case in ('contact_stain', 'clean'):
        for method in sorted({r['method'] for r in rows}):
            subset = [r for r in rows if r['case'] == case and r['method'] == method]
            print(case, method, 'core_r2', sum(r['source_core_r2'] for r in subset), '/',
                sum(r['source_core_n'] for r in subset), 'paper_ink',
                sum(r['source_remote_paper_ink'] for r in subset), '/',
                sum(r['source_remote_paper_n'] for r in subset), flush=True)


if __name__ == '__main__':
    main()
