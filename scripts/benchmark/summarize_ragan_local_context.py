"""Align already-cached 256/512 predictions; no inference or weight updates."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/benchmark'))
import run_ragan_full_continuation as full
from run_ragan_local_region_control import OUT
from compare_v5_aligned_fade_control import coverage, dump, mask, rgb, sha
from apply_ragan_real_regions import panel


def main():
    dest = OUT / 'evaluation'
    native = json.loads((dest / 'training_native_results.json').read_text())
    reviewed = json.loads((dest / 'training_reviewed_results.json').read_text())
    records = full.read_rows('train')
    reviews = full.reviewed()
    patch = '200017197_00214_1_p00'
    rows, figures = [], []
    for order in (0, 1):
        conditions = [f'baseline_order{order}', f'inkmean_order{order}', f'inkpaper_order{order}']
        for condition in conditions:
            for p, (_, trace) in reviews.items():
                for view in ('actual_batch4', 'single_crop1'):
                    row = next(r for r in native['rows'] if r['condition'] == condition and r['patch'] == p and r['view'] == view)
                    rows.append(dict(condition=condition, patch=p, view=view, **{k: row[k] for k in ('n', 'r0', 'r2')}))
                row = next(r for r in reviewed['rows'] if r['condition'] == condition and r['patch'] == p and r['family'] == 'targeted_fade')
                rows.append(dict(condition=condition, patch=p, view='full512_single', **{k: row[k] for k in ('n', 'r0', 'r2')}))
        row = next(r for r in native['rows'] if r['condition'] == conditions[0] and r['patch'] == patch and r['view'] == 'single_crop1')
        x, y = row['xy']
        record = next(r for r in records if r['patch_id'] == patch and r['damage_family'] == 'targeted_fade' and float(r['damage_strength']) == .8)
        source = rgb(ROOT / record['clean'])[y:y+256, x:x+256]
        damaged = rgb(ROOT / record['input'])[y:y+256, x:x+256]
        teacher = full.gray(ROOT / record['labels']['target'])[y:y+256, x:x+256].copy()
        valid = full.gray(ROOT / record['labels']['valid'])[y:y+256, x:x+256] > 0
        teacher[~valid] = 128
        trace = reviews[patch][1][y:y+256, x:x+256]
        assert trace.sum() == 44
        items = [('同じ原本の256窓', source), ('同じ退色.80の256窓', damaged), ('暫定教師・不明は灰', teacher)]
        for view in ('single_crop1', 'full512_single'):
            for condition, name in zip(conditions, ['補助なし', '墨芯', '墨芯＋紙']):
                if view == 'single_crop1':
                    path = dest / 'training_native_masks' / condition / (patch + '_single_crop1.png')
                    pred = mask(path)
                else:
                    path = dest / 'training_reviewed_masks' / condition / (patch + '_targeted_fade.png')
                    pred = mask(path)[y:y+256, x:x+256]
                score = coverage(pred, trace)
                expected = next(r for r in rows if r['condition'] == condition and r['patch'] == patch and r['view'] == view)
                assert all(score[k] == expected[k] for k in ('n', 'r0', 'r2'))
                size = '256' if view == 'single_crop1' else '512内の同じ窓'
                items.append((f'{name} {size}：{score["r2"]}/44', np.where(pred, 0, 255).astype(np.uint8)))
        path = dest / 'figures' / f'残存44点_推論範囲比較_順序{order+1}.png'
        panel(path, f'残存細線 / 入力範囲の影響・順序{order+1}', items)
        figures.append(str(path.relative_to(OUT)))
    dump(dest / 'context_comparison.json', dict(rows=rows, figures=figures, cached_predictions_only=True,
        same_RGB_pixels_in_aligned_window=True, changes_spatial_context_and_recomputed_auxiliaries=True,
        no_single_cause_identified=True, no_model_updates=True, formal_accuracy=False,
        source_results_sha256={name: sha(dest / name) for name in ['training_native_results.json', 'training_reviewed_results.json']},
        figure_hashes={f: sha(OUT / f) for f in figures}))
    print('Cached-context comparison:', len(rows), 'rows,', len(figures), 'PNGs')


if __name__ == '__main__':
    main()
