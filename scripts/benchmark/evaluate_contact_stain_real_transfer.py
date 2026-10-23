"""Compare contact-trained pilots on previously known real-damage development crops."""
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


def main():
    dest = OUT / 'real_transfer'
    assert not (dest / 'results.json').exists()
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3, device)
    models = {}
    for kind in ('unet', 'ragan'):
        for variant in ('original', 'contact'):
            checkpoint = OUT / f'{kind}_{variant}' / 'models' / kind / 'last.pt'
            model = make_model(kind)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
            models[f'{kind}_{variant}'] = (model.to(device).eval(), kind, sha(checkpoint))
    scoring_spec = importlib.util.spec_from_file_location('contact_real_partial', ROOT / 'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring = importlib.util.module_from_spec(scoring_spec)
    sys.modules[scoring_spec.name] = scoring
    scoring_spec.loader.exec_module(scoring)
    part = ROOT / 'tmp_work/classical_partial_labels_20261021'
    regions = json.loads((part / 'source_regions.json').read_text())
    annotations = {r['id']: r for r in json.loads((part / 'annotations_source_only.json').read_text())['regions']}
    cases = [c for c in json.loads((ROOT / 'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases']
             if c['id'] in ('D01', 'D02', 'D03', 'L07')]
    results, figures = [], []
    for case in cases:
        source = ROOT / case['source']
        assert sha(source) == case['source_sha256']
        page = rgb(source)
        x, y, w, h = case['roi']
        crop = page[y:y+h, x:x+w]
        howe_path = ROOT / 'tmp_work/classical_limit_audit_20261021/crops' / case['id'] / 'howe_fixed.png'
        predictions = {'howe_fixed': mask(howe_path)}
        assert predictions['howe_fixed'].shape == (h, w)
        items = [('原画像', crop), ('Howe固定', np.where(predictions['howe_fixed'], 0, 255).astype(np.uint8))]
        for name, (model, kind, _) in models.items():
            pred = real.infer_roi(model, kind, page, case['roi'], device) < .5
            predictions[name] = pred
            path = dest / 'masks' / case['id'] / f'{name}.png'
            save(path, pred)
            items.append((name, np.where(pred, 0, 255).astype(np.uint8)))
        figure = dest / 'figures' / f'{case["id"]}_原本比較.png'
        real.panel(figure, case['id'] + ' 実損傷 / 接触汚れ学習', items)
        figures.append(dict(case=case['id'], path=str(figure.relative_to(dest)), sha256=sha(figure)))
        for region in [r for r in regions if r['case'] == case['id']]:
            rx, ry, rw, rh = region['roi']
            labels, paths = scoring.raster(region, annotations[region['id']])
            for name, pred in predictions.items():
                p = pred[ry-y:ry-y+rh, rx-x:rx-x+rw]
                assert p.shape == labels.shape
                results.append(dict(case=case['id'], region=region['id'], method=name,
                    checkpoint_sha256=models[name][2] if name in models else None,
                    source_sha256=sha(source), **scoring.measure(p, labels, paths)))
        print(case['id'], 'evaluated', flush=True)
    dump(dest / 'results.json', dict(rows=results, figures=figures,
        known_development_images=True, provisional_partial_source_labels=True, formal_accuracy=False))
    for row in results:
        print(row['region'], row['method'], 'ink_r2', row['ink_r2'], '/', row['ink_n'],
              'nonink', row['nonink_false'], '/', row['nonink_n'], flush=True)


if __name__ == '__main__':
    main()
