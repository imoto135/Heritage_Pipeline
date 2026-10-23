"""Diagnose which fixed fusion requirements fail along D03 source-only traces."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
sys.path.insert(0, str(ROOT / 'tmp_work/makura_supervised_smoke_20261021/code_snapshot'))
from compare_v5_aligned_fade_control import sha, dump, rgb, mask
from probe_ridge_gate import features
from ragan_pilot_support import make_model
import apply_ragan_real_regions as real

OUT = ROOT / 'tmp_work/makura_v5_exposure_loss_control_20261021/d03_missing_stroke'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def panel(items, path):
    w, h, scale, gap, labelh = 150, 145, 4, 12, 27
    im = Image.new('RGB', (3 * (w * scale + gap) + gap, 2 * (h * scale + labelh + gap) + gap), '#faf9f5')
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, 17)
    for i, (title, arr) in enumerate(items):
        x = gap + i % 3 * (w * scale + gap)
        y = gap + i // 3 * (h * scale + labelh + gap)
        d.text((x, y), title, font=font, fill='#1f3642')
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        tile = Image.fromarray(arr.astype(np.uint8)).resize((w * scale, h * scale), Image.Resampling.NEAREST)
        im.paste(tile, (x, y + labelh))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.25, device)
    checkpoint = ROOT / 'tmp_work/makura_v5_exposure_loss_control_20261021/unet_v3_exposure/models/unet/last.pt'
    model = make_model('unet')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    model.to(device).eval()
    spec = importlib.util.spec_from_file_location('scoring', ROOT / 'scripts/benchmark/evaluate_classical_partial_labels.py')
    scoring = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = scoring
    spec.loader.exec_module(scoring)
    part = ROOT / 'tmp_work/classical_partial_labels_20261021'
    region = next(r for r in json.loads((part / 'source_regions.json').read_text()) if r['id'] == 'contact_upper')
    ann = next(r for r in json.loads((part / 'annotations_source_only.json').read_text())['regions'] if r['id'] == 'contact_upper')
    case = next(c for c in json.loads((ROOT / 'tmp_work/classical_limit_audit_20261021/protocol.json').read_text())['cases'] if c['id'] == 'D03')
    page = rgb(ROOT / case['source'])
    x, y, w, h = case['roi']
    q = 1 - real.infer_roi(model, 'unet', page, case['roi'], device)
    howe = mask(ROOT / 'tmp_work/classical_limit_audit_20261021/crops/D03/howe_fixed.png')
    local, ridge = features(page)
    local = local[y:y+h, x:x+w]
    ridge = ridge[y:y+h, x:x+w]
    distance = ndi.distance_transform_edt(~howe)
    rx, ry, rw, rh = region['roi']
    ox, oy = rx - x, ry - y
    rows = []
    for path in ann['ink_paths']:
        if path['confidence'] != 'high':
            continue
        pts = scoring.ordered_line(path['xy'])
        xx = pts[:, 0] + ox
        yy = pts[:, 1] + oy
        p = q[yy, xx]
        v = local[yy, xx]
        r = ridge[yy, xx]
        d = distance[yy, xx]
        candidate = p >= .2
        gate3 = candidate & (d <= 3) & (r >= 6)
        gate8 = candidate & (d <= 8) & (r >= 6) & (v >= 16)
        rows.append(dict(path=path['id'], n=len(pts),
                         howe_exact=int(howe[yy, xx].sum()), howe_within2=int((d <= 2).sum()),
                         unet_q_ge020=int(candidate.sum()), howe_distance_le3=int((d <= 3).sum()),
                         howe_distance_le8=int((d <= 8).sum()), local_contrast_ge16=int((v >= 16).sum()),
                         ridge_ge6=int((r >= 6).sum()), eligible_gate3=int(gate3.sum()),
                         eligible_gate8=int(gate8.sum()),
                         median_unet_ink_probability=float(np.median(p)),
                         median_local_contrast=float(np.median(v)),
                         median_ridge=float(np.median(r)),
                         median_howe_distance=float(np.median(d))))
    labels, _ = scoring.raster(region, ann)
    nonink = labels == 2
    q_region = q[oy:oy+rh, ox:ox+rw]
    ridge_region = ridge[oy:oy+rh, ox:ox+rw]
    howe_region = howe[oy:oy+rh, ox:ox+rw]
    distance_region = distance[oy:oy+rh, ox:ox+rw]
    nonink_metrics = dict(n=int(nonink.sum()),
        unet_q_ge020=int(((q_region >= .2) & nonink).sum()),
        median_unet_ink_probability=float(np.median(q_region[nonink])),
        ridge_ge6=int(((ridge_region >= 6) & nonink).sum()),
        howe_ink=int((howe_region & nonink).sum()),
        gate3_ink=int((((q_region >= .2) & (distance_region <= 3) & (ridge_region >= 6)) & nonink).sum()))
    # Source-only trace overlay: blue dots are annotated centers, not predicted positives.
    raw = page[ry:ry+rh, rx:rx+rw]
    trace = raw.copy()
    for path in ann['ink_paths']:
        pts = scoring.ordered_line(path['xy'])
        trace[pts[:, 1], pts[:, 0]] = [15, 100, 245]
    def gray_mask(m):
        return np.where(m, 0, 255).astype(np.uint8)
    items = [('原画像', raw), ('原画像＋追跡中心', trace),
             ('Howe', gray_mask(howe[oy:oy+rh, ox:ox+rw])),
             ('U-Net 墨確率≥0.2', gray_mask(q[oy:oy+rh, ox:ox+rw] >= .2)),
             ('ridge≥6', gray_mask(ridge[oy:oy+rh, ox:ox+rw] >= 6)),
             ('Howe＋固定ridge規則', gray_mask((howe | ((q >= .2) & (distance <= 3) & (ridge >= 6)))[oy:oy+rh, ox:ox+rw]))]
    figure = OUT / 'D03上部の欠落要因.png'
    panel(items, figure)
    dump(OUT / 'results.json', dict(rows=rows, nonink_metrics=nonink_metrics, figure=str(figure.relative_to(OUT)),
        figure_sha256=sha(figure), source_sha256=sha(ROOT / case['source']),
        annotations_sha256=sha(part / 'annotations_source_only.json'),
        checkpoint_sha256=sha(checkpoint), diagnostic_on_previously_inspected_case=True,
        formal_accuracy=False))
    for row in rows:
        print(row, flush=True)
    print('nonink', nonink_metrics, flush=True)


if __name__ == '__main__':
    main()
