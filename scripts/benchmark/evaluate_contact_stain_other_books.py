"""Source-selected other-book transfer cases: one contact mark and one isolated spot."""
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
from run_contact_stain_training_control import OUT
from evaluate_contact_stain_counterfactual import howe_mask
from compare_v5_aligned_fade_control import sha, dump, rgb, save
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

DEST = OUT / 'other_book_transfer'
CASES = ('T07', 'C01')


def freeze():
    assert not (DEST / 'protocol.json').exists()
    selected = {c['test_id']: c for c in json.loads((ROOT / 'tmp_work/npj_contact_stain_search_20261007/inference_selection.json').read_text())['cases']
                if c.get('test_id') in CASES}
    assert set(selected) == set(CASES)
    cases = []
    for key in CASES:
        row = selected[key]
        source = ROOT / row['original']
        assert sha(source) == row['original_sha256']
        crop_path = DEST / 'inputs' / f'{key}_original.png'
        assert crop_path.exists(), 'Prepare source-only crop before freezing'
        x, y, w, h = row['crop_xywh']
        page = rgb(source)
        assert np.array_equal(rgb(crop_path), page[y:y+h, x:x+w])
        cases.append(dict(id=key, book=row['book'], source=row['original'], source_sha256=sha(source),
            crop_xywh=row['crop_xywh'], original_crop_sha256=sha(crop_path)))
    assert len({c['book'] for c in cases}) == 2
    dump(DEST / 'protocol.json', dict(cases=cases, selected_before_new_model_outputs=True,
        selection='Previously source-screened T07 contact mark and C01 isolated spot, different books from training',
        T07_stroke_box_xywh=[290, 380, 150, 30], T07_stain_paper_box_xywh=[360, 305, 90, 50],
        C01_spot_box_xywh=[340, 320, 90, 120],
        T07_stroke_proxy='source grayscale<85 within box, radius-2 readout',
        paper_boxes='manual source-only regions, provisional; hidden ink possible',
        methods=['howe_fixed','sauvola_w61_k020','unet_contact_q50','ragan_contact_q50','howe_or_ragan_contact_q50'],
        same_RA_and_union_threshold_as_previous=.5, known_previous_NPJ_outputs=True,
        no_expert_GT=True, no_formal_accuracy=True))


def rectangle(box, shape):
    x, y, w, h = box
    result = np.zeros(shape, bool)
    result[y:y+h, x:x+w] = True
    return result


def main():
    protocol = json.loads((DEST / 'protocol.json').read_text())
    assert not (DEST / 'results.json').exists()
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.3, device)
    models = {}
    for kind in ('unet', 'ragan'):
        checkpoint = OUT / f'{kind}_contact/models/{kind}/last.pt'
        model = make_model(kind)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
        models[kind] = (model.to(device).eval(), sha(checkpoint))
    spec = importlib.util.spec_from_file_location('howe_other_book_contact', ROOT / 'tmp_work/howe_diagnosis_20261008/experiment.py')
    howe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = howe
    spec.loader.exec_module(howe)
    rows, figures = [], []
    for case in protocol['cases']:
        key = case['id']
        source = ROOT / case['source']
        assert sha(source) == case['source_sha256']
        page = rgb(source)
        x, y, w, h = case['crop_xywh']
        crop = page[y:y+h, x:x+w]
        assert sha(DEST / 'inputs' / f'{key}_original.png') == case['original_crop_sha256']
        howe_pred = howe_mask(crop, DEST / 'scratch' / key, 'howe', howe)
        g = howe.deep.base.gray(crop)
        predictions = {'howe_fixed': howe_pred,
            'sauvola_w61_k020': howe.deep.base.sauvola(g, 61, .2),
            'unet_contact_q50': real.infer_roi(models['unet'][0], 'unet', page, case['crop_xywh'], device) < .5,
            'ragan_contact_q50': real.infer_roi(models['ragan'][0], 'ragan', page, case['crop_xywh'], device) < .5}
        predictions['howe_or_ragan_contact_q50'] = howe_pred | predictions['ragan_contact_q50']
        if key == 'T07':
            stroke_box = rectangle(protocol['T07_stroke_box_xywh'], (h, w))
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            core = stroke_box & (gray < 85)
            paper = rectangle(protocol['T07_stain_paper_box_xywh'], (h, w))
            assert core.sum() >= 100 and paper.sum() >= 1000
        else:
            core = np.zeros((h, w), bool)
            paper = rectangle(protocol['C01_spot_box_xywh'], (h, w))
        items = [('原画像', crop)]
        for method in protocol['methods']:
            pred = predictions[method]
            path = DEST / 'masks' / key / f'{method}.png'
            save(path, pred)
            dilated = ndi.binary_dilation(pred, structure=ndi.generate_binary_structure(2, 1), iterations=2)
            rows.append(dict(case=key, book=case['book'], method=method, prediction_sha256=sha(path),
                source_stroke_proxy_n=int(core.sum()), source_stroke_proxy_r2=int((dilated & core).sum()),
                source_selected_stain_or_spot_n=int(paper.sum()),
                source_selected_stain_or_spot_ink=int((pred & paper).sum())))
            items.append((method, np.where(pred, 0, 255).astype(np.uint8)))
        figure = DEST / 'figures' / f'{key}_原本比較.png'
        real.panel(figure, key + ' / 別本の実損傷（暫定）', items)
        figures.append(dict(case=key, path=str(figure.relative_to(DEST)), sha256=sha(figure)))
        print(key, 'evaluated', flush=True)
    dump(DEST / 'results.json', dict(protocol_sha256=sha(DEST / 'protocol.json'), rows=rows,
        figures=figures, source_only_proxies=True, no_expert_GT=True, formal_accuracy=False))
    for r in rows:
        print(r['case'], r['method'], 'stroke_r2', r['source_stroke_proxy_r2'], '/',
            r['source_stroke_proxy_n'], 'stain_ink', r['source_selected_stain_or_spot_ink'], '/',
            r['source_selected_stain_or_spot_n'], flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'freeze':
        freeze()
    elif len(sys.argv) == 2 and sys.argv[1] == 'run':
        main()
    else:
        raise SystemExit('usage: ... freeze|run')
