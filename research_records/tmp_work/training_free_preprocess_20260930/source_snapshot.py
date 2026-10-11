"""Exploratory native-resolution foreground comparison; no training or manual GT.

Uses the frozen outputs of npj_preprocess_review_20260930 as references.
Sauvola outputs are foreground candidates, not black/red/stain segmentations.
All files and matplotlib caches are written beneath --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, default=ROOT/'tmp_work/npj_preprocess_review_20260930')
    parser.add_argument('--output', type=Path, default=ROOT/'tmp_work/training_free_preprocess_20260930')
    args = parser.parse_args()
    review, out = args.review.resolve(), args.output.resolve()
    if not out.is_relative_to(ROOT):
        parser.error('--output must be inside the project')
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'plot_cache')
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from skimage.filters import threshold_sauvola
    import skimage
    cv2.setNumThreads(1)
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

    def read(path, flag=cv2.IMREAD_COLOR):
        value = cv2.imread(str(path), flag)
        if value is None:
            raise ValueError(f'Cannot read {path}')
        return value

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), value):
            raise OSError(path)

    def overlay(original, mask):
        result = original.copy()
        color = np.array([225, 100, 38])  # BGR, same foreground blue in every method
        result[mask] = np.round(.45*original[mask] + .55*color).astype(np.uint8)
        return result

    def figure(images, titles, path, title):
        row_height = max(1.9, min(6., 4.5*images[0].shape[0]/images[0].shape[1]+.65))
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 2*row_height))
        for ax, value, label in zip(axes.flat, images, titles):
            ax.imshow(cv2.cvtColor(value, cv2.COLOR_BGR2RGB), interpolation='nearest')
            ax.set_title(label, fontproperties=font, fontsize=11)
            ax.axis('off')
        fig.suptitle(title, fontproperties=font, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.8, w_pad=1.5)
        fig.savefig(path, dpi=145, facecolor='white')
        plt.close(fig)

    prior = json.loads((review/'manifest.json').read_text())
    rois = json.loads((review/'rois.json').read_text())
    modes = {
        'v1_ace': '凍結v1：同じACEのCSC（黒＋赤）',
        'v3_ace_rescue': '現在v3：同じACEのCSC（黒＋赤）',
        'sauvola_raw': '原画像：Sauvola 75',
        'sauvola_local': '局所背景補正：Sauvola 75',
        'sauvola_multiscale': '局所背景補正：3尺度の多数決',
    }
    input_hashes = {}
    baseline_hashes = {}
    for case in prior['cases']:
        relative = case['original']
        assert sha(ROOT/relative) == prior['inputs'][relative], relative
        input_hashes[relative] = sha(ROOT/relative)
        completion = json.loads((review/case['id']/'complete.json').read_text())
        for mode in ('v1_ace', 'v3_ace_rescue'):
            for name in ('black.png', 'red.png'):
                relative = f"{case['id']}/{mode}/{name}"
                expected = completion['outputs'][relative]
                assert sha(review/relative) == expected, relative
                baseline_hashes[relative] = expected

    manifest = dict(
        purpose='Exploration without training/manual GT; no claim of accuracy or red/stain separation',
        settings=dict(sauvola_windows=[31, 75, 151], k=.2, r=128., comparison_window=75,
                      multiscale='at least two of three window decisions',
                      background_downsample=4, background_closing='31x31 ellipse on downsampled gray',
                      background_blur_sigma=2., normalization='clip(255*gray/max(background,gray),0,255)',
                      classification_resolution='native; downsampling is only for background estimation',
                      border_policy='no forced background for candidates; CSC retains original 4% border',
                      red_policy='no separate red decision; original RGB is preserved in all previews'),
        versions=dict(python=sys.version, opencv=cv2.__version__, skimage=skimage.__version__, numpy=np.__version__),
        source_sha256=sha(Path(__file__)), review_manifest_sha256=sha(review/'manifest.json'),
        rois_sha256=sha(review/'rois.json'), inputs=input_hashes, baseline_masks=baseline_hashes,
        modes=modes)
    manifest_path = out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Output already belongs to different code/parameters/inputs; use a new output directory')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    shutil.copyfile(__file__, out/'source_snapshot.py')
    metrics, gallery = [], []
    for case in prior['cases']:
        started = time.monotonic()
        target = out/case['id']
        target.mkdir(exist_ok=True)
        original = read(ROOT/case['original'])
        h, w = original.shape[:2]
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, ((w+3)//4, (h+3)//4), interpolation=cv2.INTER_AREA)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT)
        smooth = cv2.GaussianBlur(closed.astype(np.float32), (0, 0), 2., borderType=cv2.BORDER_REFLECT)
        background = cv2.resize(smooth, (w, h), interpolation=cv2.INTER_LINEAR)
        background = np.maximum(background, gray.astype(np.float32))
        normalized = np.round(np.clip(255.*gray/np.maximum(background, 1.), 0, 255)).astype(np.uint8)
        votes = np.zeros((h, w), np.uint8)
        local = None
        for window in (31, 75, 151):
            foreground = normalized < threshold_sauvola(normalized, window_size=window, k=.2, r=128.)
            votes += foreground
            if window == 75:
                local = foreground
        masks = {
            'sauvola_raw': gray < threshold_sauvola(gray, window_size=75, k=.2, r=128.),
            'sauvola_local': local,
            'sauvola_multiscale': votes >= 2,
        }
        for mode in ('v1_ace', 'v3_ace_rescue'):
            masks[mode] = (read(review/case['id']/mode/'black.png', 0)>127) | (
                read(review/case['id']/mode/'red.png', 0)>127)
        write(target/'original.png', original)
        write(target/'gray.png', cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        write(target/'normalized.png', cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR))
        write(target/'background_estimate.png', np.clip(np.round(background),0,255).astype(np.uint8))
        write(target/'scale_votes.png', votes*85)
        # Disagreement across scales is a diagnostic, not calibrated uncertainty.
        write(target/'scale_disagreement.png', ((votes>0)&(votes<3)).astype(np.uint8)*255)
        write(target/'ace.png', read(review/case['id']/'ace.png'))
        overlays = {}
        for mode in modes:
            mask = masks[mode]
            assert mask.shape == (h, w)
            write(target/mode/'foreground.png', mask.astype(np.uint8)*255)
            overlays[mode] = overlay(original, mask)
            write(target/mode/'overlay.png', overlays[mode])
            delta = np.zeros_like(original)
            delta[:] = (239,235,231)
            delta[mask & ~masks['v1_ace']] = (180,165,0)  # teal: added to foreground
            delta[~mask & masks['v1_ace']] = (25,135,245)  # orange: removed from foreground
            write(target/mode/'difference_v1.png', delta)
            metrics.append(dict(page=case['id'], mode=mode, foreground_fraction=float(mask.mean()),
                                added_vs_v1_pixels=int((mask & ~masks['v1_ace']).sum()),
                                removed_vs_v1_pixels=int((~mask & masks['v1_ace']).sum())))
        titles = ['原画像', 'v1 ACE CSC：前景', 'v3 同じACE：前景',
                  '原画像 Sauvola 75', '局所背景補正 Sauvola 75', '局所背景補正 3尺度多数決']
        images = [original, *[overlays[m] for m in modes]]
        figure(images, titles, target/'overview.png', case['id']+'　青＝前景候補（正解ラベルではない）')
        page = dict(id=case['id'], title=case['title'], width=w, height=h, rois=[])
        if case['id'] == '200018243_00003_2':
            page['title'] = '本文・裏写り候補・欄外の混在'
        for idx, roi in enumerate(rois[case['id']]):
            x,y,rw,rh = roi['xywh']
            assert 0<=x<x+rw<=w and 0<=y<y+rh<=h
            slices=(slice(y,y+rh),slice(x,x+rw))
            name=f'roi_{idx+1:02d}.png'
            figure([value[slices] for value in images], titles, target/name, roi['title']+'　青＝前景候補')
            page['rois'].append(dict(title=roi['title'], xywh=roi['xywh'], image=f"{case['id']}/{name}"))
            for mode in modes:
                mask=masks[mode][slices]; reference=masks['v1_ace'][slices]
                metrics.append(dict(page=case['id'], mode=mode, roi=idx+1,
                                    foreground_fraction=float(mask.mean()),
                                    added_vs_v1_pixels=int((mask & ~reference).sum()),
                                    removed_vs_v1_pixels=int((~mask & reference).sum())))
        gallery.append(page)
        print(f"{case['id']}: {w}x{h}, five foreground conditions completed in {time.monotonic()-started:.1f}s", flush=True)
    (out/'diagnostics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    template=Path(__file__).with_name('training_free_viewer.html').read_text()
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(dict(cases=gallery,modes=modes),ensure_ascii=False)))
    outputs={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*.png'))}
    (out/'complete.json').write_text(json.dumps(dict(status='complete',cases=len(gallery),
        rois=sum(len(p['rois']) for p in gallery),outputs=outputs),ensure_ascii=False,indent=2)+'\n')
    print('Comparison figures and viewer complete; no model trained and no labels created.',flush=True)


if __name__ == '__main__':
    main()
