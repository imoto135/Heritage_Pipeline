"""Apply the existing frozen v6 unchanged to original-first new-book selection."""
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import compare_saved_npj_v6 as driver
from collect_saved_npj_failures import saved_masks
from build_npj_v6_comparison import dump, figure, overlay, read, sha, write

ROOT = driver.ROOT
OUT = ROOT/'tmp_work/unseen_thin_margin_validation_20261001'


def render():
    selected = json.loads((OUT/'baseline_review.json').read_text())
    manifest = json.loads((OUT/'comparison_manifest.json').read_text())
    manifest['dependencies'][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    manifest.update(runner='scripts/benchmark/compare_unseen_thin_margin_v6.py', primary_artifact='PNG',
        new_book_selection_original_first=True, frozen_rim_source_sha256=selected['frozen_rim_source_sha256'])
    dump(OUT/'comparison_manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'png_runner_snapshot.py')
    rows = []
    for page in selected['cases']:
        key = page['page']; original = read(ROOT/page['original']); npj = saved_masks(page)
        background = read(ROOT/page['npj_background'], True)
        assert np.all(sum(m.astype(np.uint8) for m in [*npj.values(), background]) == 1)
        v6 = {k: read(OUT/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
        regions = [dict(id=key+'_whole', title='ページ全体', xywh=[0, 0, original.shape[1], original.shape[0]])] + page['rois']
        for roi in regions:
            rid = roi['id']; x, y, w, h = roi['xywh']; s = (slice(y, y+h), slice(x, x+w)); crop = original[s]
            assert crop.shape[:2] == (h, w)
            baseline = overlay(crop, [('edit', npj['edit'][s]), ('blue', npj['black'][s]), ('red', npj['red'][s])])
            proposed = overlay(crop, [('edit', v6['edit'][s]), ('hold', v6['hold'][s]), ('blue', v6['protect'][s]), ('red', v6['red'][s])])
            figure([crop, baseline, proposed], ['原画像', '保存済みnpj', '固定v6'], OUT/'classification_comparisons'/(rid+'.png'),
                roi['title']+' · '+key, '分類マスクだけの比較。青＝墨・保護、赤＝朱、黄＝編集、紫＝保留。DDRM後ではない。', max_width=620)
            if not rid.endswith('_whole'):
                native = OUT/'classification_native'/rid
                for name, value in dict(original=crop, npj_edit=npj['edit'][s], v6_edit=v6['edit'][s]).items(): write(native/(name+'.png'), value)
            rows.append(dict(id=rid, page=key, xywh=roi['xywh'], npj_edit_pixels=int(npj['edit'][s].sum()),
                v6_edit_pixels=int(v6['edit'][s].sum()), v6_protected_pixels=int(v6['protect'][s].sum())))
        del original, npj, v6, background
        gc.collect()
    for relative, digest in manifest['dependencies'].items(): assert sha(ROOT/relative) == digest
    dump(OUT/'classification_diagnostics.json', rows)
    dump(OUT/'comparison_complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p)
        for folder in ['classification_comparisons', 'classification_native', 'v6_masks'] for p in (OUT/folder).rglob('*.png')}))
    dump(OUT/'classification_verification.json', dict(status='passed', original_selection_first=True, pages=3, regions=12,
        saved_npj_not_recomputed=True, frozen_v6=True, primary_artifact='PNG', restoration=False, pixel_GT=False))
    print('Frozen v6 comparisons complete: 3 new books / 12 ROIs', flush=True)


if __name__ == '__main__':
    driver.OUT = OUT; driver.render = render
    driver.compute()
