"""Recompute local npj-style preprocessing and plot auditable visual diagnostics.

This does not alter the frozen baseline, perform restoration, or create labels.
All processing uses original-resolution pages. Display scaling is separate.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
CASES = [
    dict(id='hnsd001_011', title='水染み・紙のムラとACE',
         original='data/train/hnsd00000/hnsd001_011.jpg',
         legacy_ace='data/split_dataset_csc/train/hnsd00000/hnsd001_011.jpg'),
    dict(id='200010454_00002_2', title='朱印と茶色い汚れの分離',
         original='data/test/200010454/200010454_00002_2.jpg',
         legacy_ace='data/split_dataset_csc/test/200010454/200010454_00002_2.jpg'),
    dict(id='200017197_00008_0', title='濃淡のある筆画・細い線',
         original='data/makura_csc_mask/200017197_00008_0/00_original.png',
         legacy_ace='data/makura_csc_mask/200017197_00008_0/00_input.png'),
    dict(id='200018243_00003_2', title='本文・朱書・裏写りの混在',
         original='data/test/200018243/200018243_00003_2.jpg',
         legacy_ace=None),
]
MODES = {
    'v1_no_ace': 'v1 CSC・ACEなし',
    'v1_ace': 'v1 CSC・CPU ACEあり',
    'v3_ace_rescue': '改良v3・同じACE・明度救済あり',
}
COLORS = {'background': (231, 235, 239), 'black': (38, 100, 225),
          'red': (228, 52, 69), 'degradation': (235, 178, 23)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    return image


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(path)


def masks_from_result(result):
    return {'black': result.black_text_mask > 127,
            'red': result.red_text_mask > 127,
            'degradation': result.degradation_mask > 127,
            'background': result.background_mask > 127}


def read_masks(directory):
    return {key: cv2.imread(str(directory / f'{key}.png'), 0) == 255 for key in COLORS}


def classification(masks):
    shape = next(iter(masks.values())).shape
    rgb = np.zeros((*shape, 3), np.uint8)
    total = np.zeros(shape, np.uint8)
    for key, color in COLORS.items():
        rgb[masks[key]] = color
        total += masks[key]
    rgb[total > 1] = (175, 61, 206)
    return rgb


def panel(images, titles, path, note, columns=3):
    rows = (len(images)+columns-1)//columns
    aspect = images[0].shape[0] / images[0].shape[1]
    row_height = max(1.9, min(6.0, 4.5 * aspect + .65))
    fig, axes = plt.subplots(rows, columns, figsize=(columns*4.5, rows*row_height), squeeze=False)
    for ax, image, title in zip(axes.flat, images, titles):
        # Nearest-neighbour display preserves discrete label maps.
        if image.ndim == 2:
            ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        else:
            ax.imshow(image, interpolation='nearest')
        ax.set_title(title, fontproperties=FONT, fontsize=11)
        ax.axis('off')
    for ax in list(axes.flat)[len(images):]:
        ax.axis('off')
    fig.suptitle(note, fontproperties=FONT, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94), h_pad=2.8, w_pad=1.5)
    fig.savefig(path, dpi=145, facecolor='white')
    plt.close(fig)


def compute():
    paths = {'ace': ROOT/'modules/ddrm_npj/scripts/ace.py',
             'v1': ROOT/'modules/ddrm_npj/scripts/csc.py',
             'v3': ROOT/'modules/ddrm_npj_v3/scripts/csc.py'}
    modules = {key: load_module(f'review_{key}', path) for key, path in paths.items()}
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths.values()}
    manifest = dict(purpose='Exploratory visual review; not manual ground truth or a representative benchmark',
                    baseline='Local frozen ddrm_npj v1; exact reproduction of published results is not established',
                    ace=dict(implementation='v1 CPU; recomputed from source, native resolution',
                             alpha=5.0, n_samples=200, seed=0),
                    csc=dict(seed=42, max_samples=50000, border_margin_ratio=.04,
                             v3_ink_rescue_by_lightness=True), source_hashes=source_hashes,
                    selection='Four previously investigated diagnostic pages; selected for known failure mechanisms',
                    cases=CASES, inputs={case['original']: sha(ROOT/case['original']) for case in CASES})
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError('Existing review has different inputs or code')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for key, path in paths.items():
        destination = OUT/'source_snapshot'/key/path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(path.read_bytes())
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    metrics = []
    for case in CASES:
        target = OUT/case['id']
        target.mkdir(exist_ok=True)
        original = read(ROOT/case['original'])
        write(target/'original.png', original)
        completion = target/'complete.json'
        if completion.exists():
            record = json.loads(completion.read_text())
            assert record['signature'] == signature
            for path, expected in record['outputs'].items():
                assert sha(OUT/path) == expected, path
            metrics += record['metrics']
            print(f"{case['id']}: verified completed outputs", flush=True)
            continue
        started = time.monotonic()
        print(f"{case['id']}: CPU ACE start {original.shape}", flush=True)
        ace = modules['ace'].automatic_color_equalization(original, alpha=5., n_samples=200, seed=0)
        assert ace.shape == original.shape
        write(target/'ace.png', ace)
        print(f"{case['id']}: CPU ACE done {time.monotonic()-started:.1f}s", flush=True)
        if case['legacy_ace']:
            legacy = read(ROOT/case['legacy_ace'])
            assert legacy.shape == original.shape
            write(target/'legacy_saved_ace.png', legacy)
        case_metrics = []
        results = {}
        for mode in MODES:
            mode_started = time.monotonic()
            kwargs = dict(seed=42, max_samples=50000, border_margin_ratio=.04)
            if mode == 'v3_ace_rescue':
                result = modules['v3'].color_space_clustering(ace, ink_rescue_by_lightness=True, **kwargs)
            else:
                result = modules['v1'].color_space_clustering(original if mode == 'v1_no_ace' else ace, **kwargs)
            masks = masks_from_result(result)
            assert all(mask.shape == original.shape[:2] for mask in masks.values())
            rgb = classification(masks)
            write(target/mode/'classes.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            for key, mask in masks.items():
                write(target/mode/f'{key}.png', mask.astype(np.uint8)*255)
            count = sum(mask.astype(np.uint8) for mask in masks.values())
            entry = dict(case=case['id'], mode=mode,
                         **{f'{key}_fraction': float(mask.mean()) for key, mask in masks.items()},
                         overlap_pixels=int((count > 1).sum()), unassigned_pixels=int((count == 0).sum()),
                         wall_seconds=time.monotonic()-mode_started)
            case_metrics.append(entry)
            results[mode] = rgb
            print(json.dumps(entry, ensure_ascii=False), flush=True)
        orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        ace_rgb = cv2.cvtColor(ace, cv2.COLOR_BGR2RGB)
        panel([orig_rgb, ace_rgb, results['v1_no_ace'], results['v1_ace'],
               cv2.imread(str(target/'v1_ace/black.png'), 0), results['v3_ace_rescue']],
              ['原画像', 'CPU ACE後（v1から再計算）', 'v1分類：ACEなし', 'v1分類：ACEあり',
               'v1黒文字マスク：白が検出領域', '改良v3：同じACE＋明度救済'],
              target/'overview.png', case['id']+'　'+case['title'])
        files = sorted(path for path in target.rglob('*.png') if path.name != 'overview.png')
        completion.write_text(json.dumps(dict(signature=signature,
                              outputs={str(path.relative_to(OUT)): sha(path) for path in files},
                              metrics=case_metrics), ensure_ascii=False, indent=2))
        metrics += case_metrics
    with open(OUT/'class_fractions.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (OUT/'status.json').write_text(json.dumps(dict(status='computed', cases=len(CASES), signature=signature), indent=2))
    print('All four native-resolution preprocessing cases completed.', flush=True)


def gallery():
    rois = json.loads((OUT/'rois.json').read_text())
    cases = []
    for case in CASES:
        # Correct the exploratory page caption after inspecting the source.
        # Computation metadata and image/mask hashes remain unchanged.
        case = dict(case)
        if case['id'] == '200018243_00003_2':
            case['title'] = '本文・裏写り候補・欄外の混在'
        target = OUT/case['id']
        original = cv2.cvtColor(read(target/'original.png'), cv2.COLOR_BGR2RGB)
        ace = cv2.cvtColor(read(target/'ace.png'), cv2.COLOR_BGR2RGB)
        maps = {mode: cv2.cvtColor(read(target/mode/'classes.png'), cv2.COLOR_BGR2RGB) for mode in MODES}
        # Precompute transparent overlays so the viewer works from file://
        # without reading canvas pixels or requiring a web server.
        for mode in MODES:
            mode_masks = read_masks(target/mode)
            for selected in ['foreground', 'all', *COLORS]:
                rgba = np.zeros((*original.shape[:2], 4), np.uint8)
                if selected in COLORS:
                    mask = mode_masks[selected]
                    rgba[:, :, :3][mask] = COLORS[selected]
                else:
                    mask = np.ones(original.shape[:2], bool) if selected == 'all' else (
                        mode_masks['black'] | mode_masks['red'] | mode_masks['degradation'])
                    rgba[:, :, :3] = maps[mode]
                rgba[:, :, 3][mask] = 255
                write(target/mode/f'overlay_{selected}.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        v1 = read_masks(target/'v1_ace')
        raw = read_masks(target/'v1_no_ace')
        panel([original, ace, maps['v1_no_ace'], maps['v1_ace'],
               v1['black'].astype(np.uint8)*255, maps['v3_ace_rescue']],
              ['原画像', 'CPU ACE後（v1から再計算）', 'v1分類：ACEなし', 'v1分類：ACEあり',
               'v1黒文字マスク：白が検出領域', '改良v3：同じACE＋明度救済'],
              target/'overview.png', case['id']+'　'+case['title'])
        crops = []
        roi_metrics = []
        for idx, roi in enumerate(rois[case['id']]):
            x, y, w, h = roi['xywh']
            assert x >= 0 and y >= 0 and x+w <= original.shape[1] and y+h <= original.shape[0]
            slices = (slice(y, y+h), slice(x, x+w))
            images = [original[slices], ace[slices], maps['v1_no_ace'][slices], maps['v1_ace'][slices],
                      v1['black'][slices].astype(np.uint8)*255, maps['v3_ace_rescue'][slices]]
            name = f'roi_{idx+1:02d}.png'
            panel(images, ['原画像', 'ACE後', 'v1分類：ACEなし', 'v1分類：ACEあり',
                           'v1黒文字：白が検出領域', '改良v3：同じACE＋明度救済'], target/name,
                  roi['title']+f'　原寸ROI ({x}, {y}, {w}, {h})')
            crops.append(dict(roi, image=f"{case['id']}/{name}"))
            g = cv2.cvtColor(original[slices], cv2.COLOR_RGB2GRAY).astype(float)
            a = cv2.cvtColor(ace[slices], cv2.COLOR_RGB2GRAY).astype(float)
            roi_metrics.append(dict(title=roi['title'], xywh=roi['xywh'],
                                    original_gray_mean=float(g.mean()), ace_gray_mean=float(a.mean()),
                                    original_gray_std=float(g.std()), ace_gray_std=float(a.std()),
                                    **{f'v1_ace_{k}_fraction': float(m[slices].mean()) for k,m in v1.items()},
                                    **{f'v1_raw_{k}_fraction': float(m[slices].mean()) for k,m in raw.items()}))
        (target/'roi_statistics.json').write_text(json.dumps(roi_metrics, ensure_ascii=False, indent=2))
        cases.append(dict(case, width=original.shape[1], height=original.shape[0], rois=crops))
    data = json.dumps(dict(cases=cases, modes=MODES, colors=COLORS), ensure_ascii=False)
    template = (OUT/'viewer_template.html').read_text()
    (OUT/'index.html').write_text(template.replace('__REVIEW_DATA__', data))
    print('Gallery, original-resolution ROI figures and interactive viewer written.', flush=True)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['compute', 'gallery'])
    args = parser.parse_args()
    (compute if args.stage == 'compute' else gallery)()
