"""Validate frozen thin3.5 + single-step near8 gates on original-selected books."""
import gc
import inspect
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_paper_rim_lowfrequency_v2 as frozen
import probe_thin_margin_components as thin
import probe_anchored_thin_fragments as anchor
from build_npj_v6_comparison import dump, read, sha, write
from collect_saved_npj_failures import saved_masks
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT = frozen.ROOT
INPUT = ROOT/'tmp_work/unseen_thin_margin_validation_20261001'
OUT = INPUT/'fixed_masks_v2'
MODES = ['v6', 'rim', 'thin', 'near8']


def anchor_helper():
    source = inspect.getsource(anchor.masks_for_page)
    old = "key = page['page']; previous = prior.mask.OUT/'pages'/key"
    assert source.count(old) == 1
    source = source.replace('def masks_for_page(page, source, original):',
                            'def anchored_for_page(page, source, original, thin_folder):')
    source = source.replace(old, "key = page['page']; previous = thin_folder")
    namespace = dict(prior=anchor.prior, RADII=anchor.RADII, read=read, np=np, cv2=cv2)
    exec(compile(source, '<frozen-anchor-routing-only>', 'exec'), namespace)
    return namespace['anchored_for_page'], source


def parity(helper):
    rows = []
    sources = anchor.prior.mask.prior.MARGIN.SOURCES+[anchor.prior.mask.new.INPUT]
    for source in sources:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key = page['page']; original = read(ROOT/page['original'])
            result = helper(page, source, original, thin.OUT/'pages'/key)
            for radius in anchor.RADII:
                for name in ['edit', 'added', 'fragment_core']:
                    assert np.array_equal(result[radius][name], read(anchor.OUT/'pages'/key/('near'+str(radius))/(name+'.png'), True))
                assert result[radius]['notes'] == json.loads((anchor.OUT/'pages'/key/('near'+str(radius))/'fragments.json').read_text())
            rows.append(dict(page=key, bitwise_identical_radii=anchor.RADII))
            print('Routing parity:', key, flush=True)
            del original, result; gc.collect()
    assert len(rows) == 18
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for folder in [thin.OUT, anchor.OUT]: check_bundle(folder)
    selected = json.loads((INPUT/'baseline_review.json').read_text())
    assert not set(c['book'] for c in selected['cases']) & set(selected['excluded_books'])
    for name, digest in selected['frozen_candidate_sources'].items(): assert sha(Path(__file__).parent/name) == digest
    assert sha(Path(frozen.__file__)) == selected['frozen_rim_source_sha256']
    comparison = json.loads((INPUT/'comparison_manifest.json').read_text())
    for relative, digest in comparison['dependencies'].items(): assert sha(ROOT/relative) == digest
    for relative, digest in json.loads((INPUT/'comparison_complete.json').read_text())['outputs'].items(): assert sha(INPUT/relative) == digest
    assert sha(INPUT/'comparison_runner_snapshot.py') == comparison['source_sha256']
    helper, helper_source = anchor_helper(); reproduced = parity(helper)
    dependencies = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)), **selected['inputs']}
    for module in [frozen, thin, anchor, frozen.MARGIN, frozen.MARGIN.first, frozen.MARGIN.second,
                   frozen.prior.prior.previous.mask_probe]:
        dependencies[str(Path(module.__file__).relative_to(ROOT))] = sha(Path(module.__file__))
    for folder, names in [(INPUT, ['baseline_review.json', 'visual_selection.json', 'comparison_manifest.json', 'comparison_complete.json']),
                          (thin.OUT, ['manifest.json', 'complete.json']), (anchor.OUT, ['manifest.json', 'complete.json'])]:
        for name in names: dependencies[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    manifest = dict(dependencies=dependencies, source_sha256=sha(Path(__file__)), pages=3, regions=12,
        original_selection_first=True, excluded_books=selected['excluded_books'],
        settings=dict(thin_core_peak=3.5, anchor_distance=8, anchor_recursive=False, full_page=True,
                      ROI_label_gating=False, other_gates_unchanged=True),
        parameters_tuned_after_outputs=False, pixel_GT=False, training_independence_unknown=True)
    if (OUT/'manifest.json').exists(): assert json.loads((OUT/'manifest.json').read_text()) == manifest
    else: dump(OUT/'manifest.json', manifest)
    shutil.copyfile(__file__, OUT/'source_snapshot.py')
    (OUT/'copied_anchor_helper.py').write_text(helper_source)
    dump(OUT/'routing_parity.json', reproduced)
    pages = []; rows = []
    for page in selected['cases']:
        key = page['page']; original = read(ROOT/page['original'])
        v6 = {k: read(INPUT/'v6_masks'/key/(k+'.png'), True) for k in ['edit', 'protect', 'hold', 'red', 'extra']}
        features, _ = frozen.MARGIN.load_features(key, original, v6)
        old, old_maps, old_details = frozen.MARGIN.infer(original, v6, features)
        completions, maps, details = frozen.prior.prior.previous.mask_probe.infer(original, features, old['dark_support'], old_maps)
        complete = completions['completed']
        guard = maps['warm_anchor_guard']|maps['long_line_guard']|v6['red']|v6['extra']
        core = old_maps['accepted_core']|maps['completed_core']
        rim_edit, rim_added, excluded = frozen.rim_mask(complete['edit'], core, features['light_drop'], maps['safe_domain'], guard)
        strict, _ = thin.select(old_maps['dark_support'], old_maps['top_margin'], features['color_seed'], 6)
        assert np.array_equal(strict, old_maps['accepted_core'])
        relaxed, notes = thin.select(old_maps['dark_support'], old_maps['top_margin'], features['color_seed'], 3.5)
        new_core = relaxed&~strict
        halo = cv2.dilate(new_core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        thin_added = halo&maps['safe_domain']&~guard&(features['light_drop'] >= 2)&~rim_edit
        dest = OUT/'pages'/key; thin_folder = dest/'thin_input'
        for name, value in dict(edit=rim_edit|thin_added, new_core=new_core, domain=maps['safe_domain'], guard=guard).items():
            write(thin_folder/(name+'.png'), value)
        anchored = helper(page, INPUT, original, thin_folder)[8]
        values = dict(v6=v6)
        for mode, edit in [('rim', rim_edit), ('thin', rim_edit|thin_added), ('near8', anchored['edit'])]:
            added = edit&~v6['edit']
            values[mode] = dict(edit=edit, added=added, protect=v6['protect']&~added,
                                hold=v6['hold']&~added, red=v6['red'], extra=v6['extra'])
            assert not np.any(added&(guard|~maps['safe_domain']))
            assert not np.any(edit&(values[mode]['protect']|values[mode]['hold']|v6['red']|v6['extra']))
        for mode, masks in values.items():
            for name, value in masks.items(): write(dest/mode/(name+'.png'), value)
        for name, value in {**maps, 'dark_support': old_maps['dark_support'], 'top_margin': old_maps['top_margin'],
                                'strict_core': strict, 'relaxed_core': relaxed, 'new_core': new_core, 'rim_guard': guard,
                                'rim_added': rim_added, 'thin_added': thin_added, 'fragment_added': anchored['added']}.items():
            write(dest/'features'/(name+'.png'), value)
        np.save(dest/'features/light_drop.npy', features['light_drop'], allow_pickle=False)
        dump(dest/'components.json', dict(margin=old_details, completion=details, thin=notes, near8=anchored['notes']))
        pages.append(dict(page=key, rule_y=details['rule_y'], margin_added=int(old['dark_support']['added'].sum()),
                          completed_added=int(complete['added'].sum()), rim_added=int(rim_added.sum()),
                          thin_added=int(thin_added.sum()), near8_added=int(anchored['added'].sum()), thin_components=notes,
                          near8_components=anchored['notes']))
        npj = saved_masks(page)
        regions = [dict(id=key+'_whole', title='ページ全体', xywh=[0, 0, original.shape[1], original.shape[0]])]+page['rois']
        for roi in regions:
            rid = roi['id']; x, y, w, h = roi['xywh']; s = (slice(y, y+h), slice(x, x+w)); crop = original[s]
            paints = [frozen.MARGIN.first.paint(crop, {k: v[s] for k, v in values[m].items()}) for m in MODES]
            npj_image = frozen.MARGIN.first.overlay(crop, [('edit', npj['edit'][s]), ('blue', npj['black'][s]), ('red', npj['red'][s])])
            figure([crop, npj_image]+paints, ['原画像', '保存済みnpj', '固定v6', '固定の輪郭回収', '薄いコア3.5', '一段近接8px'],
                   OUT/'comparisons'/(rid+'.png'), roi['title']+' · '+key,
                   '黄＝編集、青＝保護、赤＝印影、紫＝保留。推定マスクの比較、復元なし。',
                   '原画像だけで3冊を先に選択。条件の再調整なし。損傷の正解ラベルはない。')
            add = values['near8']['edit']&~v6['edit']; preview = crop.copy(); preview[v6['edit'][s]] = (0,200,255); preview[add[s]] = (255,120,0)
            figure([crop, preview], ['原画像', '黄＝v6編集、青＝今回の追加'], OUT/'additions'/(rid+'.png'),
                   roi['title']+' · 追加領域', f'v6からの追加 {int(add[s].sum()):,}画素。青は追加の意味。', '復元なし。')
            rows.append(dict(id=rid, page=key, title=roi['title'], xywh=roi['xywh'],
                             diagnostics={m: dict(edit=int(values[m]['edit'][s].sum()), added_vs_v6=int((values[m]['edit'][s]&~v6['edit'][s]).sum())) for m in MODES},
                             npj_edit=int(npj['edit'][s].sum()), thin_added=int(thin_added[s].sum()), near8_added=int(anchored['added'][s].sum())))
        print('Frozen thin+near8:', key, pages[-1], flush=True)
        del original, v6, features, old, old_maps, maps, complete, strict, relaxed, new_core, halo, values, npj, anchored
        gc.collect()
    assert len(pages) == 3 and len(rows) == 15
    for relative, digest in dependencies.items(): assert sha(ROOT/relative) == digest
    dump(OUT/'diagnostics.json', dict(pages=pages, regions=rows))
    dump(OUT/'verification.json', dict(status='passed', pages=3, regions=12, full_page=True,
                                     routing_parity_pages=18, strict6_parity=True, guards_preserved=True,
                                     parameters_tuned_after_outputs=False, restoration=False, pixel_GT=False))
    dump(OUT/'complete.json', dict(outputs={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*')
                                          if p.is_file() and p.suffix in ['.png', '.npy', '.json', '.py'] and p.name not in ['manifest.json', 'complete.json']}))
    print('Complete: fixed thin+near8 on 3 new books /12 ROIs', flush=True)


if __name__ == '__main__': main()
