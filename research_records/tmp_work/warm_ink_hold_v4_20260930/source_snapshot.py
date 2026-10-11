"""Second warm-ink development trial: restrict routing and added hold.

Uses frozen v2 decisions and v3 input-derived color candidates. No training,
GT, ink relabelling, restoration, or alteration of preceding experiments.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from explore_warm_ink_hold import ROOT, V2, POS, get_cases, sha, dump, read, write, distance_to
import cv2
import numpy as np

OUT = ROOT / 'tmp_work/warm_ink_hold_v4_20260930'
V3 = ROOT / 'tmp_work/warm_ink_hold_v3_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
KEYS = ['red', 'protect', 'hold', 'edit_allowed', 'stain_candidate',
        'dense_stain', 'dense_hold', 'protection_guard']
MODES = ['red_touch', 'red_protect_context']
SETTINGS = dict(
    color='Reuse frozen v3 contextual_candidate; same input-derived color direction and page gate (>=20 anchors)',
    routing='Color candidate intersected with v2 protect | hold | stain_candidate; exclude dense_stain | dense_hold',
    connectivity=dict(closing_ellipse=5, original_color_pixels_only=True, connectivity=8),
    red_touch=dict(minimum_confirmed_red_pixels_per_component=3, no_area_cap=True),
    nearby_protect=dict(minimum_existing_protected_warm_pixels=3,
                        maximum_minimum_red_distance=512., minimum_bridge_area=12,
                        maximum_fill=.75, maximum_aspect=35., no_area_cap=True),
    added_hold='Only accepted color pixels already in v2 edit_allowed',
    policy='Confirmed red, protection, stain candidates and guard stay frozen; only hold expands and edit shrinks',
    limitation='Development heuristics; nearby protected warm ink may include stains; no independent accuracy claim')


def candidates(color, enabled, ref):
    routing = color & (ref['protect'] | ref['hold'] | ref['stain_candidate'])
    routing &= ~(ref['dense_stain'] | ref['dense_hold'])
    bridged = cv2.morphologyEx(routing.astype(np.uint8), cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bridged |= routing.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
    red_support = np.bincount(labels[(routing & ref['red'])], minlength=n)
    protected_support = np.bincount(labels[(routing & ref['protect'] & ~ref['red'])], minlength=n)
    minimum_red_distance = np.full(n, np.inf, np.float32)
    # Use original routing pixels for distance, not the artificial bridges.
    d = distance_to(ref['red'])
    np.minimum.at(minimum_red_distance, labels[routing], d[routing])
    cw = stats[:, cv2.CC_STAT_WIDTH]; ch = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    fill = area / np.maximum(cw.astype(float) * ch, 1.)
    aspect = np.maximum(cw / np.maximum(ch, 1.), ch / np.maximum(cw, 1.))
    shape = (area >= 12) & (fill <= .75) & (aspect <= 35.)
    touching = (red_support >= 3) & enabled
    nearby = shape & (protected_support >= 3) & (minimum_red_distance <= 512.) & enabled
    touching[0] = False; nearby[0] = False
    accepted = dict(red_touch=touching, red_protect_context=touching | nearby)
    extras = {key: routing & value[labels] & ref['edit_allowed'] for key, value in accepted.items()}
    records = []
    for i in np.flatnonzero(touching | nearby):
        x, y, w, h, component_area = map(int, stats[i])
        records.append(dict(id=int(i), xywh=[x, y, w, h], bridge_area=component_area,
                            red_support_pixels=int(red_support[i]), protected_support_pixels=int(protected_support[i]),
                            minimum_red_distance=float(minimum_red_distance[i]),
                            red_touch=bool(touching[i]), nearby_protect=bool(nearby[i])))
    return extras, routing, dict(context_enabled=enabled, routing_pixels=int(routing.sum()),
                                red_touch_components=int(touching.sum()),
                                nearby_protect_components=int(nearby.sum()), accepted_components=records)


def overlay(original, mask, color):
    result = original.copy()
    result[mask] = np.round(.45 * original[mask] + .55 * np.array(color)).astype(np.uint8)
    return result


def verification():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['source_sha256'] == sha(Path(__file__))
    assert manifest['settings'] == SETTINGS
    for rel, expected in manifest['dependencies'].items():
        assert sha(ROOT / rel) == expected, rel
    complete = json.loads((OUT / 'complete.json').read_text())
    for rel, expected in complete['outputs'].items():
        assert sha(OUT / rel) == expected, rel
    negative = []
    for row in manifest['cases']:
        dest = OUT / row['id']; refdir = ROOT / row['reference']
        original = read(ROOT / row['original'])
        assert np.array_equal(original, read(dest / 'original.png'))
        ref = {key: read(refdir / (key + '.png'), True) for key in KEYS}
        for key in ['red', 'protect', 'stain_candidate']:
            assert np.array_equal(ref[key], read(dest / (key + '.png'), True))
        for mode in MODES:
            extra = read(dest / (mode + '_extra_hold.png'), True)
            hold = read(dest / (mode + '_hold.png'), True)
            edit = read(dest / (mode + '_edit_allowed.png'), True)
            assert not (extra & ~ref['edit_allowed']).any()
            assert np.array_equal(hold, ref['hold'] | extra)
            assert np.array_equal(edit, ref['edit_allowed'] & ~extra)
            assert not (edit & ~ref['stain_candidate']).any()
            for excluded in [hold, ref['protect'], ref['red'], ref['protection_guard']]:
                assert not (edit & excluded).any()
            if row['group'] == 'negative_regression':
                assert not extra.any() and np.array_equal(edit, ref['edit_allowed'])
        if row['group'] == 'negative_regression':
            negative.append(row['id'])
        for p in dest.glob('*.png'):
            if p.name == 'original.png':
                continue
            value = cv2.imread(str(p), 0)
            assert value.shape == original.shape[:2] and set(np.unique(value)).issubset({0, 255})
        # Verify actual native crop pixels, including overlay colors.
        masks = dict(v2_edit=ref['edit_allowed'],
                     v3_edit=read(V3 / row['id'] / 'context_warm_hold_edit_allowed.png', True),
                     red_touch_edit=read(dest / 'red_touch_edit_allowed.png', True),
                     context_edit=read(dest / 'red_protect_context_edit_allowed.png', True),
                     context_extra=read(dest / 'red_protect_context_extra_hold.png', True))
        for i, roi in enumerate(row['rois'], 1):
            x, y, w, h = roi['xywh']; sl = (slice(y, y+h), slice(x, x+w))
            assert np.array_equal(read(dest / 'review' / f'roi_{i:02d}_original.png'), original[sl])
            for key, mask in masks.items():
                color = (176, 65, 155) if key == 'context_extra' else (23, 178, 235)
                assert np.array_equal(read(dest / 'review' / f'roi_{i:02d}_{key}.png'),
                                      overlay(original[sl], mask[sl], color))
    x, y, w, h = (461, 591, 84, 55)
    dense = read(V2 / 'hnsd001_011/dense_stain.png', True)[y:y+h, x:x+w]
    edit = read(OUT / 'hnsd001_011/red_protect_context_edit_allowed.png', True)[y:y+h, x:x+w]
    assert int(dense.sum()) == 3131 and int((dense & edit).sum()) == 3131
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ('href', 'src') and value and not value.startswith('#'):
                    assert (OUT / value).is_file(), value
    Links().feed((OUT / 'index.html').read_text())
    subprocess.run(['node', '--check', str(OUT / 'viewer.js')], check=True, cwd=ROOT)
    result = dict(status='passed', pages=21, books=20, fixed_review_rois=52,
                  original_pixels_unchanged=True, frozen_inputs_and_code_checked=True,
                  output_hashes_checked=len(complete['outputs']), native_crops_and_overlays_checked=True,
                  red_protect_stain_unchanged=True, extra_hold_subset_of_old_edit=True,
                  edit_excludes_hold_protect_guard_red=True, negative_pages_unchanged=negative,
                  known_dense_stain_still_editable_pixels=3131,
                  viewer_links_and_javascript_checked=True, browser_rendering_tested=False,
                  neural_training=False, pixel_GT=False, independent_accuracy_test=False)
    dump(OUT / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    cases = get_cases()
    v3_manifest = json.loads((V3 / 'manifest.json').read_text())
    assert cases == v3_manifest['cases']
    v3_outputs = json.loads((V3 / 'complete.json').read_text())['outputs']
    dependencies = dict(v3_manifest['dependencies'])
    for p in [V3 / 'manifest.json', V3 / 'complete.json',
              Path(__file__).with_name('explore_warm_ink_hold.py'), POS / 'failure_zoom_locations.json']:
        dependencies[str(p.relative_to(ROOT))] = sha(p)
    for row in cases:
        for key in ['contextual_candidate.png', 'features.json', 'context_warm_hold_edit_allowed.png']:
            p = V3 / row['id'] / key; expected = v3_outputs[str(p.relative_to(V3))]
            assert sha(p) == expected
            dependencies[str(p.relative_to(ROOT))] = expected
    for rel, expected in dependencies.items():
        assert sha(ROOT / rel) == expected, rel
    manifest = dict(source_sha256=sha(Path(__file__)), settings=SETTINGS, cases=cases,
                    dependencies=dependencies, purpose='Second development trial; 21 known pages, not independent test',
                    neural_training=False, pixel_GT=False, independent_accuracy_test=False)
    mp = OUT / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError('Use a new directory for changed settings, code or inputs')
    dump(mp, manifest); shutil.copyfile(__file__, OUT / 'source_snapshot.py')
    names = dict(original='原画像', v2_edit='v2：編集許可', v3_edit='試作v3：編集許可',
                 red_touch_edit='朱に接する成分だけ：編集許可', context_edit='既存保護も手掛かり：編集許可',
                 context_extra='既存保護も手掛かり：追加保留')
    colors = dict(edit=(23, 178, 235), hold=(176, 65, 155))
    metrics = []; cards = []; gallery = []
    for row in cases:
        start = time.monotonic(); stem = row['id']; dest = OUT / stem
        original = read(ROOT / row['original']); h, w = original.shape[:2]
        ref = {key: read(ROOT / row['reference'] / (key + '.png'), True) for key in KEYS}
        prior = read(V3 / stem / 'context_warm_hold_edit_allowed.png', True)
        color = read(V3 / stem / 'contextual_candidate.png', True)
        enabled = json.loads((V3 / stem / 'features.json').read_text())['context_enabled']
        extras, routing, feat = candidates(color, enabled, ref)
        write(dest / 'original.png', original); write(dest / 'routing_candidate.png', routing)
        for key in ['red', 'protect', 'stain_candidate']:
            write(dest / (key + '.png'), ref[key])
        dump(dest / 'features.json', feat)
        edits = {}
        for mode in MODES:
            extra = extras[mode]; edits[mode] = ref['edit_allowed'] & ~extra
            write(dest / (mode + '_extra_hold.png'), extra)
            write(dest / (mode + '_hold.png'), ref['hold'] | extra)
            write(dest / (mode + '_edit_allowed.png'), edits[mode])
        images = dict(original=original, v2_edit=overlay(original, ref['edit_allowed'], colors['edit']),
                      v3_edit=overlay(original, prior, colors['edit']),
                      red_touch_edit=overlay(original, edits['red_touch'], colors['edit']),
                      context_edit=overlay(original, edits['red_protect_context'], colors['edit']),
                      context_extra=overlay(original, extras['red_protect_context'], colors['hold']))
        parts = []; page_gallery = dict(id=stem, rois=[])
        rects = [(None, 'ページ全体', [0, 0, w, h])] + [(i, r['title'], r['xywh']) for i, r in enumerate(row['rois'], 1)]
        for index, title, box in rects:
            x, y, rw, rh = box; sl = (slice(y, y+rh), slice(x, x+rw))
            entry = dict(page=stem, book=row['book'], group=row['group'], roi=index, title=title, xywh=box,
                         v2_edit_pixels=int(ref['edit_allowed'][sl].sum()), v3_edit_pixels=int(prior[sl].sum()))
            for mode in MODES:
                entry[mode + '_edit_pixels'] = int(edits[mode][sl].sum())
                entry[mode + '_extra_hold_pixels'] = int(extras[mode][sl].sum())
            metrics.append(entry)
            if index is None:
                continue
            panels = []; paths = {}
            fig, axes = plt.subplots(2, 3, figsize=(13.5, 2*max(2., min(6., 4.5*rh/rw+.65))))
            for ax, (key, label) in zip(axes.flat, names.items()):
                rel = f'{stem}/review/roi_{index:02d}_{key}.png'; write(OUT / rel, images[key][sl]); paths[key] = rel
                panels.append(f'<figure data-mode="{key}"><figcaption>{html.escape(label)}</figcaption><a href="{rel}"><img loading="lazy" src="{rel}"></a></figure>')
                ax.imshow(cv2.cvtColor(images[key][sl], cv2.COLOR_BGR2RGB), interpolation='nearest')
                ax.set_title(label, fontproperties=font, fontsize=10); ax.axis('off')
            fig.suptitle(stem + ' · ' + title + '（開発資料・未注釈）', fontproperties=font, fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .95))
            rel = f'{stem}/review/roi_{index:02d}_comparison.png'; fig.savefig(OUT / rel, dpi=145, facecolor='white'); plt.close(fig)
            parts.append(f'<section><h3>{html.escape(title)} · {box}</h3><p>v2編集許可 {entry["v2_edit_pixels"]:,} → '
                         f'今回 {entry["red_protect_context_edit_pixels"]:,}画素。減少は編集見送りで、文字精度ではない。</p>'
                         f'<p><a href="{rel}">比較図</a></p><div class="grid">{"".join(panels)}</div></section>')
            page_gallery['rois'].append(dict(title=title, xywh=box, comparison=rel, images=paths))
        cards.append(f'<article id="{stem}"><h2>{stem}</h2><p>冊 {row["book"]} · {row["group"]} · 原寸 {w}×{h}</p>{"".join(parts)}</article>')
        gallery.append(page_gallery)
        full = next(m for m in reversed(metrics) if m['page'] == stem and m['roi'] is None)
        print(f'{stem}: edit {full["v2_edit_pixels"]}->{full["red_protect_context_edit_pixels"]}, '
              f'extra hold={full["red_protect_context_extra_hold_pixels"]}, {time.monotonic()-start:.1f}s', flush=True)
        dump(OUT / 'progress.json', dict(status='running', completed_pages=len(gallery), total_pages=21))
    dump(OUT / 'diagnostics.json', metrics); dump(OUT / 'gallery.json', gallery)
    zooms = []
    for i, row in enumerate(json.loads((POS / 'failure_zoom_locations.json').read_text()), 1):
        stem = row['page']; x, y, w, h = row['xywh']; sl = (slice(y, y+h), slice(x, x+w))
        refrow = next(r for r in cases if r['id'] == stem)
        original = read(OUT / stem / 'original.png')[sl]
        masks = dict(v2_edit=read(ROOT / refrow['reference'] / 'edit_allowed.png', True)[sl],
                     v3_edit=read(V3 / stem / 'context_warm_hold_edit_allowed.png', True)[sl],
                     red_touch_edit=read(OUT / stem / 'red_touch_edit_allowed.png', True)[sl],
                     context_edit=read(OUT / stem / 'red_protect_context_edit_allowed.png', True)[sl],
                     context_extra=read(OUT / stem / 'red_protect_context_extra_hold.png', True)[sl])
        panels = []
        for key in ['original', 'v2_edit', 'v3_edit', 'red_touch_edit', 'context_edit', 'context_extra']:
            picture = original if key == 'original' else overlay(original, masks[key], colors['hold' if key == 'context_extra' else 'edit'])
            write(OUT / 'failure_zooms' / f'zoom_{i:02d}_{key}.png', picture)
            panel = np.full((h+34, w, 3), 255, np.uint8); panel[34:] = picture
            cv2.putText(panel, key, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1); panels.append(panel)
        path = OUT / 'failure_zooms' / f'zoom_{i:02d}_comparison.png'; write(path, np.concatenate(panels, axis=1))
        zooms.append(dict(page=stem, xywh=row['xywh'], image=str(path.relative_to(OUT)),
                          **{key + '_pixels': int(mask.sum()) for key, mask in masks.items()}, pixel_GT=False))
    dump(OUT / 'failure_zoom_diagnostics.json', zooms)
    controls = ''.join(f'<label><input type="checkbox" data-toggle="{key}" {"checked" if key in ["original", "v2_edit", "context_edit", "context_extra"] else ""}>{html.escape(label)}</label>' for key, label in names.items())
    toc = ' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in cases)
    zoom_links = ' · '.join(f'<a href="{z["image"]}">失敗箇所の拡大{i}</a>' for i, z in enumerate(zooms, 1))
    (OUT / 'viewer.js').write_text("const toggles=[...document.querySelectorAll('[data-toggle]')];function update(){for(const cb of toggles){for(const fig of document.querySelectorAll('figure[data-mode=\"'+cb.dataset.toggle+'\"]'))fig.hidden=!cb.checked;}}toggles.forEach(cb=>cb.addEventListener('change',update));document.querySelector('#native').addEventListener('change',e=>document.body.classList.toggle('native',e.target.checked));update();\n")
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>薄い朱・赤褐色の編集前保留：試作v4</title>
<style>body{font-family:system-ui,sans-serif;max-width:1800px;margin:24px auto;padding:0 18px;background:#f3f1ed;color:#202326;line-height:1.7}header,article,.controls{background:white;padding:20px;border-radius:10px;margin-bottom:22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}figure{margin:0}figure img{width:100%;height:auto}figcaption{margin:8px 0}figure[hidden]{display:none}section{border-top:1px solid #ddd;padding:16px 0}.controls label{display:inline-block;margin:6px 18px 6px 0}a{color:#075ea0}nav{overflow-wrap:anywhere}body.native .grid{display:flex;flex-wrap:nowrap;overflow:auto;align-items:flex-start}body.native figure{flex:none}body.native figure img{width:auto;max-width:none}</style>
<header><h1>薄い朱・赤褐色の編集前保留：試作v4</h1><p>21ページ・20冊・52か所。全て開発／回帰資料。黄＝編集許可、紫＝今回の追加保留。朱の確定・文字保護・汚れ候補はv2のまま。局所色候補の連結を既存の保護・保留・汚れ候補内で確認し、編集候補内だけ追加保留する。</p><p>学習・手動画素GT・原画像の削除・拡散復元は実施していない。保留は文字の認定ではなく、編集見送りである。編集面積の減少を精度改善と呼ばない。</p><p><a href="report.md">結果と条件</a> · <a href="verification.json">検証</a> · <a href="../warm_ink_hold_v3_20260930/visual_findings.md">前試作の目視所見</a> · <a href="../preprocessing_status_before_warm_rescue_20260930.md">着手前の現状</a></p><p>'''
        + zoom_links + '</p><nav>' + toc + '</nav></header><div class="controls">' + controls
        + '<p><label><input type="checkbox" id="native">原寸表示（横スクロール）</label></p></div>' + ''.join(cards) + '<script src="viewer.js"></script></html>\n')
    full = [m for m in metrics if m['roi'] is None]
    summary = dict(pages=21, books=20, rois=52, v2_edit_pixels=sum(m['v2_edit_pixels'] for m in full),
                   v3_edit_pixels=sum(m['v3_edit_pixels'] for m in full),
                   **{key + '_pixels': sum(m[key + '_pixels'] for m in full)
                      for mode in MODES for key in [mode + '_edit', mode + '_extra_hold']})
    dump(OUT / 'summary.json', summary)
    table = '\n'.join(f'| {m["page"]} | {m["v2_edit_pixels"]:,} | {m["v3_edit_pixels"]:,} | {m["red_touch_edit_pixels"]:,} | {m["red_protect_context_edit_pixels"]:,} |' for m in full)
    ztable = '\n'.join(f'| [拡大{i}：{z["page"]}]({z["image"]}) | {z["v2_edit_pixels"]:,} | {z["v3_edit_pixels"]:,} | {z["red_touch_edit_pixels"]:,} | {z["context_edit_pixels"]:,} |' for i, z in enumerate(zooms, 1))
    (OUT / 'report.md').write_text(f'''# 薄い朱・赤褐色の編集前保留：試作v4

[現状整理](../preprocessing_status_before_warm_rescue_20260930.md)のあと、[試作v3の失敗](../warm_ink_hold_v3_20260930/visual_findings.md)を踏まえ、学習なしで保留を修正した。現在の21ページ・20冊・52か所は全て開発／回帰資料である。画素GT、独立精度評価、原画像の変更、拡散復元は行っていない。

## 変更と比較

局所暖色・朱の色方向・朱の種が20以上というページ条件はv3と同じ保存済み候補を使う。連結を調べる対象を、v2の文字保護・保留・汚れ候補の和集合内に制限する。既知の濃い汚れ・濃い保留を除く。5px楕円closingは接続確認だけで、出力は元の色条件にある画素に限定する。

①同じ成分に確定朱が3画素以上ある場合は、面積上限で除外しない。②比較として、既存文字保護にある暖色画素が3以上で確定朱までの成分最短距離512px以内の成分も使う。この②の追加だけは面積12以上・充填率0.75以下・縦横比35以下を要求する。成分が巨大なだけで確定朱につながる薄い印影を落とさず、離れた筆画は既存保護を手掛かりとする。これらは開発用の経験則で、確定インクの証明ではない。

追加保留はv2で編集許可だった画素だけ。朱・文字保護・汚れ候補・ガードを変更しない。紙面全体に新しい保留を広げることは避けたが、編集可能な汚れを保留する誤りは起こり得る。保留の少なさ自体を精度改善の根拠としない。②が①よりよいか、副作用を目視する。

## ページ全体の編集候補数

| ページ | v2 | 試作v3 | ①朱に接する成分 | ②既存保護も手掛かり |
|---|---:|---:|---:|---:|
{table}

## 前回と同じ失敗箇所

| 診断拡大 | v2 | 試作v3 | ①朱に接する成分 | ②既存保護も手掛かり |
|---|---:|---:|---:|---:|
{ztable}

数値は矩形内の編集候補数で、文字Recallではない。画素GTなしの目視所見は別の `visual_findings.md` に記録する。今回の入力と出力を見たため、これらのページを未見の評価資料として使わない。

## 検証と次の判断

原画像と既存出力のハッシュ、原寸画素、二値マスク、元の朱・保護・汚れ候補の不変、追加保留が旧編集候補の部分集合であること、編集候補が保護・保留・ガードと重ならないことを確認する。同じ52矩形の原寸切り出しとオーバーレイ画素も検証する。赤のない8ページの不変と、既知の濃い斑点3,131画素が編集候補に残ることを回帰確認する。ビューアのリンクとJavaScript構文を検証し、ブラウザ実表示は未検証。

確定朱のないページ、朱から離れた孤立筆記、候補の色方向と異なる赤を救済する設計ではない。既存保護の誤りや朱に接する茶色汚れは誤保留の原因になる。汚れの編集対象率と文字誤消去を同時に見る評価は今後必要である。CSC／ACEや原論文に対する全般的な優越は未確認。試作結果を目視して、採用・追加修正・保留を判断する。
''')
    dump(OUT / 'verification.json', dict(status='pending'))
    products = [p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts
                and p.name not in ['complete.json', 'verification.json', 'progress.json', 'run.log']]
    dump(OUT / 'complete.json', dict(status='complete', outputs={str(p.relative_to(OUT)): sha(p) for p in sorted(products)}))
    verification()
    dump(OUT / 'progress.json', dict(status='verified', completed_pages=21, total_pages=21))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verification() if args.verify_only else main()
