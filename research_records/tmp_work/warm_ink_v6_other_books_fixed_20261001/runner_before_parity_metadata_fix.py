"""Apply frozen v2/v4/v5/v6 to the six reserved other-book pages.

Diagnostic comparison only. Preserve the allocation, native pixels and method
constants. Reuse the earlier computations, including the verified v2 AST adapter.
"""
from __future__ import annotations

import argparse
import gc
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/warm_ink_v6_other_books_fixed_20261001'
QUEUE = ROOT / 'tmp_work/warm_ink_v6_next_fixed_inputs_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')

import cv2
import numpy as np
import check_red_positive_fixed as adapter
import prepare_training_free_fixed_check as foreground
import explore_warm_ink_hold as v3
import refine_warm_ink_hold as v4
import compare_warm_stroke_hold as v5
import compare_split_warm_hold as v6

cv2.setNumThreads(1)
read, write, dump, sha = v3.read, v3.write, v3.dump, v3.sha
NAMES = dict(original='原画像', v2_edit='v2：編集候補', v4_edit='v4：編集候補',
             v5_edit='v5：編集候補', v6_edit='v6：編集候補',
             v6_extra='v2から増えた保留', regained='v5から増えた保留',
             red='確定朱候補', protect='v2の保護')
MASK_FILES = dict(v2_edit='v2/edit_allowed.png', v4_edit='v4_edit_allowed.png',
                  v5_edit='v5_edit_allowed.png', v6_edit='v6_edit_allowed.png',
                  v6_extra='v6_extra_hold.png', regained='regained_hold.png',
                  red='v2/red.png', protect='v2/protect.png')


def frozen_dependencies():
    dependencies = {}
    for module, directory in [(v3, v3.OUT), (v4, v4.OUT), (v5, v5.OUT), (v6, v6.OUT)]:
        manifest = json.loads((directory / 'manifest.json').read_text())
        path = Path(module.__file__)
        assert sha(path) == manifest['source_sha256'], path
        assert module.SETTINGS == manifest['settings'], path
        dependencies[str(path.relative_to(ROOT))] = sha(path)
        dependencies[str((directory / 'manifest.json').relative_to(ROOT))] = sha(directory / 'manifest.json')
    freeze = json.loads((v6.OUT / 'candidate_freeze.json').read_text())
    assert freeze['variant'] == 'split_direction' and freeze['settings'] == v6.SETTINGS
    assert freeze['source_sha256'] == sha(Path(v6.__file__))
    selection = json.loads((QUEUE / 'selection.json').read_text())
    for rel, digest in selection['dependencies'].items():
        assert sha(ROOT / rel) == digest, rel
        dependencies[rel] = digest
    for p in [QUEUE / 'selection.json', OUT / 'input_review/manifest.json',
              OUT / 'input_review/rois.json', OUT / 'input_review/findings.md',
              Path(adapter.__file__), Path(foreground.__file__)]:
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    for directory, name, field in [(adapter.FROZEN, 'refine_training_free_protection.py', 'source_sha256'),
                                  (adapter.COLOR, 'compare_training_free_color_shape.py', 'source_sha256'),
                                  (adapter.PREPROCESS, 'compare_training_free_preprocessing.py', 'source_sha256')]:
        prior = json.loads((directory / 'manifest.json').read_text())
        p = ROOT / 'scripts/benchmark' / name
        assert sha(p) == prior[field], p
        dependencies[str(p.relative_to(ROOT))] = sha(p)
        dependencies[str((directory / 'manifest.json').relative_to(ROOT))] = sha(directory / 'manifest.json')
    return selection, dependencies


def warm_masks(original, ref, weak_red):
    colors, features = v3.hold_candidates(original, ref, weak_red)
    extras4, routing, features4 = v4.candidates(colors['contextual_candidate'], features['context_enabled'], ref)
    masks5, features5 = v5.make_masks(original, ref, colors['contextual_candidate'], features,
                                     extras4['red_protect_context'])
    masks6 = v6.make_masks(ref['edit_allowed'], masks5['color_direction'], colors['contextual_candidate'],
                           masks5['near_red'], extras4['red_protect_context'],
                           masks5['ridge_weak_seed'], masks5['ridge_strong_seed'], masks5['stroke_context'])
    return colors, features, extras4, routing, features4, masks5, features5, masks6


def check_adapter_parity(refine):
    rows = []
    for group, pilot in [(adapter.COLOR, adapter.PREPROCESS),
                         (ROOT / 'tmp_work/training_free_fixed_check_20260930',
                          ROOT / 'tmp_work/training_free_fixed_check_20260930/pilot')]:
        for old in sorted(group.iterdir()):
            if not (old / 'color_shape/red.png').exists():
                continue
            masks, _ = adapter.refine_inputs(refine, group, pilot, old.name)
            for key, mask in masks.items():
                assert np.array_equal(mask, read(adapter.FROZEN / old.name / (key + '.png'), True)), (old.name, key)
            rows.append(dict(page=old.name, masks_equal=len(masks)))
            print('v2 adapter parity:', old.name, flush=True)
            del masks
            gc.collect()
    assert len(rows) == 12
    saved = read(adapter.PREPROCESS / 'hnsd001_011/original.png')
    for value, name in zip(foreground.foreground_inputs(saved),
                           ['sauvola_local/foreground.png', 'scale_votes.png', 'normalized.png']):
        assert np.array_equal(value, cv2.imread(str(adapter.PREPROCESS / 'hnsd001_011' / name), 0)), name
    # Check the actual callable chain on positive failures and a negative page.
    pages = ['200021086_00032_2', '200021763_00002_2', '200010454_00002_2', 'hnsd001_011']
    cases = {r['id']: r for r in json.loads((v6.OUT / 'manifest.json').read_text())['cases']}
    warm = []
    for stem in pages:
        row = cases[stem]
        ref = {k: read(ROOT / row['reference'] / (k + '.png'), True)
               for k in [*v4.KEYS, 'robust_seed']}
        values = warm_masks(read(ROOT / row['original']), ref,
                            read(ROOT / row['color'] / 'red_weak.png', True))
        colors, features, extras4, _, _, masks5, _, masks6 = values
        assert np.array_equal(colors['contextual_candidate'], read(v3.OUT / stem / 'contextual_candidate.png', True))
        assert features == json.loads((v3.OUT / stem / 'features.json').read_text())
        assert np.array_equal(extras4['red_protect_context'], read(v4.OUT / stem / 'red_protect_context_extra_hold.png', True))
        for key in ['color_direction', 'ridge_weak_seed', 'ridge_strong_seed', 'near_red']:
            assert np.array_equal(masks5[key], read(v5.OUT / stem / (key + '.png'), True)), key
        assert np.array_equal(masks5['stroke_context'], read(v5.OUT / stem / 'stroke_context_extra_hold.png', True))
        assert np.array_equal(masks6['split_direction'], read(v6.OUT / stem / 'split_direction_extra_hold.png', True))
        warm.append(stem)
        print('v3/v4/v5/v6 chain parity:', stem, flush=True)
        del ref, values, colors, extras4, masks5, masks6
        gc.collect()
    dump(OUT / 'adapter_parity.json', dict(status='passed', v2_pages=rows, warm_chain_pages=warm,
                                          foreground_adapter_parity=True))


def prepare():
    selection, dependencies = frozen_dependencies()
    rois = json.loads((OUT / 'input_review/rois.json').read_text())
    cases = selection['cases']
    assert len(cases) == len({r['book'] for r in cases}) == 6
    assert not ({r['book'] for r in cases} & set(selection['protocol']['exclude_book_ids']))
    input_manifest = json.loads((OUT / 'input_review/manifest.json').read_text())
    assert [r['id'] for r in cases] == [r['id'] for r in input_manifest['cases']]
    manifest = dict(purpose='Frozen other-book diagnostic comparison, purposive selection, no pixel GT',
                    source_sha256=sha(Path(__file__)), cases=cases, dependencies=dependencies,
                    rois=rois, settings=dict(v3=v3.SETTINGS, v4=v4.SETTINGS, v5=v5.SETTINGS, v6=v6.SETTINGS),
                    method_thresholds_changed=False, neural_training=False, pixel_GT=False,
                    restoration_performed=False, versions=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__))
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError('Different experiment exists; use a new output version')
    dump(path, manifest)
    shutil.copyfile(__file__, OUT / 'runner_source_snapshot.py')
    pilot = OUT / 'pilot'
    dump(pilot / 'manifest.json', dict(settings=json.loads((adapter.PREPROCESS / 'manifest.json').read_text())['settings'],
                                      inputs=input_manifest['inputs']))
    for row in cases:
        assert sha(ROOT / row['original']) == row['original_sha256']
        original = read(ROOT / row['original'])
        assert original.shape[:2] == (row['height'], row['width'])
        dest = pilot / row['id']
        done = dest / 'complete.json'
        if done.exists():
            for rel, digest in json.loads(done.read_text())['outputs'].items():
                assert sha(pilot / rel) == digest
            continue
        local, votes, normalized = foreground.foreground_inputs(original)
        for value, name in [(original, 'original.png'), (local, 'sauvola_local/foreground.png'),
                            (votes, 'scale_votes.png'), (normalized, 'normalized.png')]:
            write(dest / name, value)
        dump(done, dict(outputs={str(p.relative_to(pilot)): sha(p) for p in sorted(dest.rglob('*.png'))}))
        print('Prepared native foreground:', row['id'], flush=True)
    dump(pilot / 'complete.json', dict(outputs={str(p.relative_to(pilot)): sha(p) for p in sorted(pilot.rglob('*.png'))}))
    color_path = OUT / 'color'
    if not (color_path / 'complete.json').exists():
        subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/benchmark/compare_training_free_color_shape.py'),
                        '--pilot', str(pilot), '--review', str(OUT / 'input_review'), '--output', str(color_path)], check=True, cwd=ROOT)
    color_complete = json.loads((color_path / 'complete.json').read_text())
    for rel, digest in color_complete['outputs'].items():
        assert sha(color_path / rel) == digest, rel
    assert json.loads((color_path / 'manifest.json').read_text())['parameters'] == json.loads((adapter.COLOR / 'manifest.json').read_text())['parameters']
    return manifest


def render_review(manifest):
    rows, cards = [], []
    colors = dict(edit=(23,178,235), hold=(176,65,155), red=(69,52,228), protect=(225,100,38))
    for row in manifest['cases']:
        stem = row['id']; dest = OUT / stem
        original = read(ROOT / row['original'])
        masks = {k: read(dest / rel, True) for k, rel in MASK_FILES.items()}
        parts = []
        rectangles = [(None, 'ページ全体', [0,0,row['width'],row['height']])] + [
            (i, r['title'], r['xywh']) for i, r in enumerate(manifest['rois'][stem], 1)]
        for i, title, box in rectangles:
            x,y,w,h = box; sl = (slice(y,y+h), slice(x,x+w))
            record = dict(page=stem, book=row['book'], roi=i, title=title, xywh=box,
                          **{k+'_pixels': int(v[sl].sum()) for k,v in masks.items()})
            rows.append(record)
            folder = 'whole' if i is None else f'roi_{i:02d}'
            panels = []
            for key, label in NAMES.items():
                value = original[sl] if key == 'original' else v4.overlay(
                    original[sl], masks[key][sl], colors[key if key in ['red','protect'] else 'hold' if key in ['v6_extra','regained'] else 'edit'])
                rel = f'{stem}/review/{folder}_{key}.png'
                write(OUT / rel, value)
                panels.append(f'<figure data-mode="{key}"><figcaption>{label}</figcaption><a href="{rel}"><img src="{rel}" loading="lazy" alt="{label}"></a></figure>')
            parts.append(f'<section><h3>{html.escape(title)} · {box}</h3><p>編集候補画素：v2 {record["v2_edit_pixels"]:,} / v4 {record["v4_edit_pixels"]:,} / v5 {record["v5_edit_pixels"]:,} / v6 {record["v6_edit_pixels"]:,}。精度・誤消去率ではありません。</p><div class="grid">{"".join(panels)}</div></section>')
        cards.append(f'<article id="{stem}"><h2>{stem}</h2><p>冊 {row["book"]} · {row["width"]}×{row["height"]}</p>{"".join(parts)}</article>')
    dump(OUT / 'diagnostics.json', rows)
    full = [r for r in rows if r['roi'] is None]
    dump(OUT / 'summary.json', dict(pages=6, books=6, rois=len(rows)-6, page_totals=full,
                                    pixel_GT=False, method_thresholds_changed=False))
    shutil.copyfile(v6.OUT / 'viewer.js', OUT / 'viewer.js')
    controls = ''.join(f'<label><input type="checkbox" data-toggle="{k}" {"checked" if k in ["original","v2_edit","v6_edit","v6_extra"] else ""}>{v}</label>' for k,v in NAMES.items())
    toc = ' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in manifest['cases'])
    (OUT / 'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定v6の別冊確認</title><style>body{font-family:system-ui,sans-serif;margin:24px;background:#faf8f3;line-height:1.6}.controls{position:sticky;top:0;background:#fff;padding:12px;z-index:2}label{display:inline-block;margin-right:16px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0}figure[hidden]{display:none}img{width:100%;height:auto}.native .grid{display:flex;overflow-x:auto}.native figure{flex:0 0 auto}.native img{width:auto;max-width:none}section{margin:24px 0}article{margin:44px 0}</style><h1>固定v6の別冊確認</h1><p>既知開発20冊と重複しない6冊・6ページ・入力から選んだ21矩形。表紙2枚・本文2枚・朱印を含む2枚。色量による目的抽出で、代表標本・画素GTではありません。黄＝編集候補、紫＝追加保留。保留は文字認定ではありません。</p><p><a href="report.md">目視結果と次の判断</a> · <a href="verification.json">検証</a> · <a href="adapter_parity.json">既存計算との一致</a> · <a href="summary.json">集計</a> · <a href="input_review/findings.md">出力を見る前の所見</a></p><nav>'+toc+'</nav><div class="controls">'+controls+'<label><input type="checkbox" id="native">原寸表示</label></div>'+''.join(cards)+'<script src="viewer.js"></script></html>\n')


def verify():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['source_sha256'] == sha(Path(__file__))
    for rel, digest in manifest['dependencies'].items():
        assert sha(ROOT / rel) == digest, rel
    complete = json.loads((OUT / 'complete.json').read_text())
    for rel, digest in complete['outputs'].items():
        assert sha(OUT / rel) == digest, rel
    assert json.loads((OUT / 'adapter_parity.json').read_text())['status'] == 'passed'
    no_context = []
    overlay_colors = dict(edit=(23,178,235), hold=(176,65,155), red=(69,52,228), protect=(225,100,38))
    for row in manifest['cases']:
        stem = row['id']; dest = OUT / stem
        assert sha(ROOT / row['original']) == row['original_sha256']
        original = read(ROOT / row['original'])
        assert np.array_equal(original, read(OUT / 'pilot' / stem / 'original.png'))
        ref = {k: read(dest / 'v2' / (k+'.png'), True) for k in v4.KEYS}
        extra5 = read(dest / 'v5_extra_hold.png', True)
        extra6 = read(dest / 'v6_extra_hold.png', True)
        assert not (extra5 & ~extra6).any()
        for mode in ['v4','v5','v6']:
            extra = read(dest / (mode+'_extra_hold.png'), True)
            edit = read(dest / (mode+'_edit_allowed.png'), True)
            held = read(dest / (mode+'_hold.png'), True)
            assert not (extra & ~ref['edit_allowed']).any()
            assert np.array_equal(edit, ref['edit_allowed'] & ~extra)
            assert np.array_equal(held, ref['hold'] | extra)
            for excluded in [held,ref['red'],ref['protect'],ref['protection_guard']]:
                assert not (edit & excluded).any()
            assert not (edit & ~ref['stain_candidate']).any()
        features = json.loads((dest / 'features.json').read_text())
        if not features['v3']['context_enabled']:
            assert not extra6.any() and not extra5.any()
            assert not read(dest / 'v4_extra_hold.png', True).any()
            no_context.append(stem)
        for p in list((dest / 'v2').glob('*.png')) + list(dest.glob('*.png')):
            value = cv2.imread(str(p),0)
            assert value.shape == original.shape[:2] and set(np.unique(value)).issubset({0,255}), p
        masks = {k: read(dest / rel, True) for k,rel in MASK_FILES.items()}
        rectangles = [('whole',[0,0,row['width'],row['height']])] + [(f'roi_{i:02d}',r['xywh']) for i,r in enumerate(manifest['rois'][stem],1)]
        for folder, box in rectangles:
            x,y,w,h = box; sl = (slice(y,y+h),slice(x,x+w))
            assert np.array_equal(read(dest / 'review' / (folder+'_original.png')), original[sl])
            for key, mask in masks.items():
                color = overlay_colors[key if key in ['red','protect'] else 'hold' if key in ['v6_extra','regained'] else 'edit']
                assert np.array_equal(read(dest / 'review' / (folder+'_'+key+'.png')), v4.overlay(original[sl],mask[sl],color))
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['href','src'] and value and not value.startswith('#'):
                    assert (OUT / value).is_file(), value
    Links().feed((OUT / 'index.html').read_text())
    subprocess.run(['node','--check',str(OUT / 'viewer.js')],check=True,cwd=ROOT)
    result = dict(status='passed', pages=6, books=6, rois=21, development_book_overlap=[],
                  allocation_and_native_originals_preserved=True, hashes_checked=len(complete['outputs']),
                  original_crops_and_overlays_checked=True, native_binary_masks=True,
                  extra_hold_subset_of_v2_edit=True, v5_hold_subset_of_v6_hold=True,
                  edit_excludes_hold_red_protect_guard=True, context_disabled_pages_unchanged=no_context,
                  thresholds_changed=False, pixel_GT=False, neural_training=False, restoration=False,
                  viewer_links_and_javascript_checked=True, browser_rendering_tested=False)
    dump(OUT / 'verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def seal():
    files = [p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts
             and p.name not in ['complete.json','verification.json','progress.json','run.log']]
    dump(OUT / 'complete.json', dict(status='complete',outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(files)}))
    verify()
    dump(OUT / 'progress.json',dict(status='verified',completed_pages=6,total_pages=6))


def main():
    manifest = prepare()
    adapter.OUT = OUT
    refine = adapter.refinement_function()
    if not (OUT / 'adapter_parity.json').exists():
        check_adapter_parity(refine)
    for index,row in enumerate(manifest['cases'],1):
        started = time.monotonic(); stem = row['id']; dest = OUT / stem
        ref, components = adapter.refine_inputs(refine, OUT / 'color', OUT / 'pilot', stem)
        for key, value in ref.items():
            write(dest / 'v2' / (key+'.png'), value)
        dump(dest / 'v2/red_components.json',components)
        original = read(ROOT / row['original'])
        values = warm_masks(original, ref, read(OUT / 'color' / stem / 'red_weak.png', True))
        colors, features3, extras4, routing, features4, masks5, features5, masks6 = values
        for name,mask in [('contextual_candidate',colors['contextual_candidate']), ('routing_candidate',routing),
                          ('color_direction',masks5['color_direction']), ('near_red',masks5['near_red']),
                          ('ridge_weak_seed',masks5['ridge_weak_seed']), ('ridge_strong_seed',masks5['ridge_strong_seed']),
                          ('regained_hold',masks6['regained_hold'])]:
            write(dest / (name+'.png'),mask)
        for mode,extra in [('v4',extras4['red_protect_context']),('v5',masks5['stroke_context']),('v6',masks6['split_direction'])]:
            write(dest / (mode+'_extra_hold.png'),extra)
            write(dest / (mode+'_edit_allowed.png'),ref['edit_allowed'] & ~extra)
            write(dest / (mode+'_hold.png'),ref['hold'] | extra)
        dump(dest / 'features.json',dict(v3=features3,v4=features4,v5=features5,frozen_v5_parity=True))
        print(f'{stem}: v2 edit {int(ref["edit_allowed"].sum()):,}, v6 extra {int(masks6["split_direction"].sum()):,}; {time.monotonic()-started:.1f}s',flush=True)
        dump(OUT / 'progress.json',dict(status='running',completed_pages=index,total_pages=6))
        del original, ref, values, colors, extras4, routing, masks5, masks6
        gc.collect()
    render_review(manifest)
    if not (OUT / 'report.md').exists():
        (OUT / 'report.md').write_text('# 固定v6の別冊確認\n\n6冊・6ページの計算を完了。原寸目視と次の判断は未記録。候補数は精度ではない。\n')
    seal()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--seal',action='store_true',help='Hash and verify outputs after adding visual findings')
    args = parser.parse_args()
    verify() if args.verify_only else seal() if args.seal else main()
