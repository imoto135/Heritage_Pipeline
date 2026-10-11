"""Apply frozen margin, top-completion and rim gates on four new books."""
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import probe_paper_rim_lowfrequency_v2 as frozen
from build_npj_v6_comparison import dump, read, sha, write
from collect_saved_npj_failures import saved_masks
from package_paper_blending_review import check_bundle

ROOT = frozen.ROOT
INPUT = ROOT/'tmp_work/unseen_margin_validation_20261001'
OUT = INPUT/'fixed_masks'
MODES = ['v6', 'completed', 'rim']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # v6 comparison uses its established runner snapshot name, unlike probes.
    comparison = json.loads((INPUT/'comparison_manifest.json').read_text())
    for relative, digest in comparison['dependencies'].items(): assert sha(ROOT/relative) == digest
    for relative, digest in json.loads((INPUT/'comparison_complete.json').read_text())['outputs'].items():
        assert sha(INPUT/relative) == digest
    assert sha(INPUT/'comparison_runner_snapshot.py') == comparison['source_sha256']
    selected = json.loads((INPUT/'baseline_review.json').read_text())
    assert sha(Path(frozen.__file__)) == selected['frozen_rim_source_sha256']
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        str(Path(frozen.__file__).relative_to(ROOT)): sha(Path(frozen.__file__)),
        str((INPUT/'baseline_review.json').relative_to(ROOT)): sha(INPUT/'baseline_review.json')}
    for name in ['comparison_manifest.json', 'comparison_complete.json']:
        dependencies[str((INPUT/name).relative_to(ROOT))] = sha(INPUT/name)
    for module in [frozen.MARGIN, frozen.MARGIN.first, frozen.MARGIN.second, frozen.prior.prior.previous.mask_probe]:
        dependencies[str(Path(module.__file__).relative_to(ROOT))] = sha(Path(module.__file__))
    manifest = dict(dependencies=dependencies, source_sha256=sha(Path(__file__)), pages=4, regions=12,
        original_selection_first=True, fixed_rim_source=selected['frozen_rim_source_sha256'],
        parameters_tuned_after_outputs=False, pixel_GT=False, training_independence_unknown=True)
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    page_rows = []; rows = []
    for page in selected['cases']:
        key = page['page']; original = read(ROOT/page['original'])
        v6 = {k: read(INPUT/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
        features, _ = frozen.MARGIN.load_features(key, original, v6)
        old, old_maps, old_details = frozen.MARGIN.infer(original, v6, features)
        completions, maps, details = frozen.prior.prior.previous.mask_probe.infer(original, features, old['dark_support'], old_maps)
        complete = completions['completed']
        guard = maps['warm_anchor_guard'] | maps['long_line_guard'] | v6['red'] | v6['extra']
        core = old_maps['accepted_core'] | maps['completed_core']
        edit, added, excluded = frozen.rim_mask(complete['edit'], core, features['light_drop'], maps['safe_domain'], guard)
        rim = dict(edit=edit, added=added, protect=complete['protect']&~added,
                   hold=complete['hold']&~added, red=complete['red'])
        values = dict(v6=v6, margin=old['dark_support'], completed=complete, rim=rim)
        dest = OUT/'pages'/key
        for mode, masks in values.items():
            for name, value in masks.items(): write(dest/mode/(name+'.png'), value)
        for name, value in dict(**maps, accepted_core=old_maps['accepted_core'], rim_core=core, rim_guard=guard, **excluded).items():
            write(dest/'features'/(name+'.png'), value)
        np.save(dest/'features/light_drop.npy', features['light_drop'], allow_pickle=False)
        dump(dest/'margin_components.json', old_details)
        dump(dest/'completion_components.json', details)
        page_rows.append(dict(page=key, rule_y=details['rule_y'], margin_added=int(old['dark_support']['added'].sum()),
            completed_added=int(complete['added'].sum()), rim_added=int(added.sum()),
            accepted_margin_components=sum(c['accepted'] for c in old_details['components']),
            warm_anchor_guard_pixels=int(maps['warm_anchor_guard'].sum())))
        npj = saved_masks(page)
        regions = [dict(id=key+'_whole', title='ページ全体', xywh=[0, 0, original.shape[1], original.shape[0]])] + page['rois']
        for roi in regions:
            rid = roi['id']; x, y, w, h = roi['xywh']; s = (slice(y, y+h), slice(x, x+w)); crop = original[s]
            image_npj = frozen.MARGIN.first.overlay(crop, [('edit', npj['edit'][s]), ('blue', npj['black'][s]), ('red', npj['red'][s])])
            paints = [frozen.MARGIN.first.paint(crop, {k: v[s] for k, v in values[m].items()}) for m in MODES]
            frozen.MARGIN.first.figure([crop, image_npj]+paints,
                ['原画像', '保存済みnpj', '固定v6', '全成分回収', '固定の輪郭回収'],
                OUT/'comparisons'/(rid+'.png'), roi['title']+' · '+key,
                f'余白追加 {int(old["dark_support"]["added"][s].sum()):,} / 上端追加 {int(complete["added"][s].sum()):,} / 輪郭追加 {int(added[s].sum()):,}画素。復元なし。')
            rows.append(dict(id=rid, page=key, title=roi['title'], xywh=roi['xywh'],
                margin_added=int(old['dark_support']['added'][s].sum()), completed_added=int(complete['added'][s].sum()),
                rim_added=int(added[s].sum()), npj_edit=int(npj['edit'][s].sum()), v6_edit=int(v6['edit'][s].sum()),
                completed_edit=int(complete['edit'][s].sum()), rim_edit=int(edit[s].sum()),
                domain_pixels=int(maps['safe_domain'][s].sum()), warm_guard_pixels=int(maps['warm_anchor_guard'][s].sum())))
            if not rid.endswith('_whole'):
                for name, value in dict(original=crop, npj_edit=npj['edit'][s], v6_edit=v6['edit'][s],
                    completed_edit=complete['edit'][s], rim_edit=edit[s], rim_added=added[s],
                    domain=maps['safe_domain'][s], warm_guard=maps['warm_anchor_guard'][s]).items():
                    write(OUT/'native'/rid/(name+'.png'), value)
        print('Fixed unseen gates:', key, page_rows[-1], flush=True)
        del original, v6, features, old, old_maps, maps, complete, core, edit, added, rim, values, npj
        gc.collect()
    for rel, digest in dependencies.items(): assert sha(ROOT/rel) == digest
    dump(OUT/'diagnostics.json', dict(pages=page_rows, regions=rows))
    dump(OUT/'verification.json', dict(status='passed', pages=4, regions=12, original_selection_first=True,
        old_parameters_fixed=True, full_page_masks=True, ROI_gating=False, restoration=False, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png', '.npy', '.json'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Fixed new-book mask validation complete', flush=True)


if __name__ == '__main__': main()
