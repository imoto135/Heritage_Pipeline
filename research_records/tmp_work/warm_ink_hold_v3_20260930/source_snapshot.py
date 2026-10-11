"""Development experiment: hold faint warm ink before allowing stain edits.

No trained model, pixel GT, restoration or changes to frozen experiments.
The latest positive examples now belong to development, not independent test.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tmp_work/warm_ink_hold_v3_20260930'
V2 = ROOT / 'tmp_work/training_free_protection_v2_20260930'
POS = ROOT / 'tmp_work/red_positive_fixed_check_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')

import cv2
import numpy as np
cv2.setNumThreads(1)

SETTINGS = dict(
    purpose='Development of extra editing hold; not classification accuracy',
    paper=dict(downsample=4, closing_ellipse=31, blur_sigma=2., channelwise=True,
               rule='estimated local bright field capped below by observed channel; not a physical paper measurement'),
    chromatic=dict(classification_median=3, green_minus_red_OD_min=.012,
                   blue_minus_red_OD_min=.012, anchor_difference_min=.03,
                   minimum_anchor_pixels=20, anchor_angle_tolerance_degrees=25.,
                   formula='OD_c = log((bright_field_c + 1)/(median_input_c + 1))'),
    shape=dict(bridge_closing_ellipse=5, min_area=12, max_area=100000,
               max_fill=.75, max_aspect=35., dense_exclusion='exclude saved dense-stain and dense-hold pixels'),
    graph=dict(maximum_component_gap=64., maximum_rounds=6,
               maximum_initial_anchor_distance=512.,
               rule='component minimum distance; use only original chromatic pixels, not the artificial bridges'),
    policy='Add hold only; keep confirmed red, protection, stain candidates and guard unchanged',
    ablations=['weak_red_hold: saved weak-red color condition only',
               'global_warm_hold: eligible local chromatic components without red context',
               'context_warm_hold: eligible components linked to robust confirmed-red anchors'],
    caveat='Fixed first trial heuristics on development pages. Extra hold may include stains near red ink.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path, mask=False):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if mask else cv2.IMREAD_COLOR)
    if value is None:
        raise OSError(path)
    return value > 0 if mask else value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if value.dtype == bool:
        value = value.astype(np.uint8) * 255
    assert cv2.imwrite(str(path), value), path


def distance_to(mask):
    if not mask.any():
        return np.full(mask.shape, np.inf, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def get_cases():
    rows = []
    sources = [
        (ROOT / 'tmp_work/npj_preprocess_review_20260930', ROOT / 'tmp_work/training_free_color_shape_20260930', 'initial_development'),
        (ROOT / 'tmp_work/training_free_fixed_check_20260930/input_review', ROOT / 'tmp_work/training_free_fixed_check_20260930', 'negative_regression')]
    for review, color, group in sources:
        prior = json.loads((review / 'manifest.json').read_text())
        rois = json.loads((review / 'rois.json').read_text())
        for row in prior['cases']:
            stem = row['id']
            book = '200017197' if stem == '200017197_00008_0' else Path(row['original']).parent.name
            rows.append(dict(id=stem, original=row['original'], book=book, group=group,
                             reference=str((V2 / stem).relative_to(ROOT)), color=str((color / stem).relative_to(ROOT)),
                             rois=rois[stem]))
    for row in json.loads((POS / 'manifest.json').read_text())['cases']:
        rows.append(dict(id=row['id'], original=row['original'], book=row['book'], group='positive_development',
                         reference=str((POS / 'refined' / row['id']).relative_to(ROOT)),
                         color=str((POS / 'color' / row['id']).relative_to(ROOT)),
                         rois=[dict(title=r['name'], xywh=r['xywh']) for r in row['rois']]))
    assert len(rows) == 21 and len({r['book'] for r in rows}) == 20
    assert sum(len(r['rois']) for r in rows) == 52
    return rows


def hold_candidates(original, reference, weak_red):
    h, w = original.shape[:2]
    filtered = cv2.medianBlur(original, SETTINGS['chromatic']['classification_median'])
    small = cv2.resize(filtered, ((w+3)//4, (h+3)//4), interpolation=cv2.INTER_AREA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31,31))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REFLECT)
    bright = cv2.GaussianBlur(closed.astype(np.float32), (0,0), 2., borderType=cv2.BORDER_REFLECT)
    bright = cv2.resize(bright, (w,h), interpolation=cv2.INTER_LINEAR)
    bright = np.maximum(bright, filtered.astype(np.float32))
    od = np.log((bright + 1.) / (filtered.astype(np.float32) + 1.))
    green = od[:,:,1] - od[:,:,2]
    blue = od[:,:,0] - od[:,:,2]
    warm = (green >= .012) & (blue >= .012)
    dense = reference['dense_stain'] | reference['dense_hold']
    warm &= ~dense
    anchors = reference['red'] & reference['robust_seed'] & (green >= .03) & (blue >= .03)
    anchor_count = int(anchors.sum())
    angle = np.arctan2(blue, green)
    center = float(np.median(angle[anchors])) if anchor_count >= 20 else None
    # The global comparator deliberately omits context and page-specific angle.
    candidates = warm if center is None else warm & (np.abs(angle-center) <= np.deg2rad(25.))

    def label_components(mask):
        bridged = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))
        # Closing may remove isolated edge pixels: include the original candidates.
        bridged |= mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
        area = stats[:,cv2.CC_STAT_AREA]
        cw = stats[:,cv2.CC_STAT_WIDTH]; ch = stats[:,cv2.CC_STAT_HEIGHT]
        fill = area / np.maximum(cw.astype(float)*ch, 1.)
        aspect = np.maximum(cw / np.maximum(ch,1.), ch / np.maximum(cw,1.))
        eligible = (area>=12) & (area<=100000) & (fill<=.75) & (aspect<=35.)
        eligible[0] = False
        return n, labels, stats, eligible

    n_global, global_labels, _, global_eligible = label_components(warm)
    global_hold = warm & global_eligible[global_labels] & ~reference['protect']
    del global_labels
    n, labels, stats, eligible = label_components(candidates)
    accepted = np.zeros(n, bool)
    rounds = []
    component_rows = []
    if center is not None:
        anchor_distance = distance_to(anchors)
        minimum_anchor_distance = np.full(n, np.inf, np.float32)
        np.minimum.at(minimum_anchor_distance, labels.ravel(), anchor_distance.ravel())
        eligible &= minimum_anchor_distance <= 512.
        frontier = anchors.copy()
        for step in range(6):
            d = distance_to(frontier)
            minimum = np.full(n, np.inf, np.float32)
            np.minimum.at(minimum, labels.ravel(), d.ravel())
            added = eligible & ~accepted & (minimum <= 64.)
            rounds.append(dict(round=step+1, added_components=int(added.sum())))
            if not added.any():
                break
            accepted |= added
            frontier |= candidates & accepted[labels]
        for i in np.flatnonzero(accepted):
            x,y,cw,ch,area = map(int,stats[i])
            component_rows.append(dict(id=int(i),xywh=[x,y,cw,ch],bridge_area=area,
                                       minimum_anchor_distance=float(minimum_anchor_distance[i])))
    context_hold = candidates & accepted[labels] & ~reference['protect']
    masks = dict(warm_candidate=warm, contextual_candidate=candidates, warm_anchor=anchors,
                 weak_red_hold=weak_red & ~reference['protect'] & ~dense,
                 global_warm_hold=global_hold, context_warm_hold=context_hold)
    features = dict(anchor_pixels=anchor_count, median_anchor_angle_degrees=None if center is None else float(np.rad2deg(center)),
                    context_enabled=center is not None, accepted_components=len(component_rows),
                    rounds=rounds, accepted_component_locations=component_rows,
                    global_components=n_global-1, contextual_components=n-1)
    return masks, features


def verify():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert sha(Path(__file__)) == manifest['source_sha256']
    assert manifest['settings'] == SETTINGS
    for rel, expected in manifest['dependencies'].items():
        assert sha(ROOT / rel) == expected, rel
    complete = json.loads((OUT / 'complete.json').read_text())
    for rel, expected in complete['outputs'].items():
        assert sha(OUT / rel) == expected, rel
    negative_unchanged = []
    for row in manifest['cases']:
        stem = row['id']; dest = OUT / stem; ref = ROOT / row['reference']
        original = read(ROOT / row['original'])
        assert np.array_equal(original, read(dest / 'original.png'))
        protect = read(dest / 'protect.png', True); red = read(dest / 'red.png', True)
        assert np.array_equal(protect,read(ref / 'protect.png',True))
        assert np.array_equal(red,read(ref / 'red.png',True))
        stain = read(dest / 'stain_candidate.png', True); guard = read(ref / 'protection_guard.png', True)
        old_hold = read(ref / 'hold.png', True); old_edit = read(ref / 'edit_allowed.png', True)
        assert np.array_equal(stain,read(ref / 'stain_candidate.png', True))
        for mode in ['weak_red_hold','global_warm_hold','context_warm_hold']:
            extra = read(dest / (mode+'.png'), True)
            held = read(dest / (mode+'_hold.png'), True)
            edit = read(dest / (mode+'_edit_allowed.png'), True)
            assert np.array_equal(held, old_hold | extra)
            assert np.array_equal(edit, old_edit & ~extra)
            for excluded in [held,protect,guard,red]:
                assert not (edit & excluded).any()
            assert not (edit & ~stain).any()
        for p in dest.glob('*.png'):
            if p.name == 'original.png':
                continue
            value = cv2.imread(str(p),0)
            assert value.shape == original.shape[:2] and set(np.unique(value)).issubset({0,255})
        if row['group'] == 'negative_regression':
            assert not read(dest / 'context_warm_hold.png', True).any()
            assert np.array_equal(read(dest / 'context_warm_hold_edit_allowed.png', True), old_edit)
            negative_unchanged.append(stem)
    # The known dense stain remains editable; it is not a GT-labelled benchmark.
    x,y,w,h = (461,591,84,55)
    old = read(V2 / 'hnsd001_011/dense_stain.png', True)[y:y+h,x:x+w]
    edited = read(OUT / 'hnsd001_011/context_warm_hold_edit_allowed.png', True)[y:y+h,x:x+w]
    assert int(old.sum()) == 3131 and int((old & edited).sum()) == 3131
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            for key,value in attrs:
                if key in ('href','src') and value and not value.startswith('#'):
                    assert (OUT / value).is_file(), value
    Links().feed((OUT / 'index.html').read_text())
    subprocess.run(['node','--check',str(OUT / 'viewer.js')],check=True,cwd=ROOT)
    result = dict(status='passed',development_pages=21,books=20,fixed_review_rois=52,
                  output_hashes_checked=len(complete['outputs']),original_pixels_unchanged=True,
                  confirmed_red_unchanged=True,protect_unchanged=True,stain_candidate_unchanged=True,
                  edit_is_subset_of_v2=True,edit_excludes_protect_hold_guard_red=True,
                  negative_context_results_unchanged=negative_unchanged,known_dense_stain_still_editable_pixels=3131,
                  viewer_links_and_javascript_checked=True,browser_rendering_tested=False,
                  neural_training=False,pixel_GT=False,independent_accuracy_test=False)
    dump(OUT / 'verification.json', result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    cases = get_cases()
    hashes = {}
    # Check every consumed frozen output using the prior completion records.
    frozen_outputs = json.loads((V2 / 'complete.json').read_text())['outputs']
    positive_outputs = json.loads((POS / 'complete.json').read_text())['outputs']
    color_roots = [ROOT / 'tmp_work/training_free_color_shape_20260930',
                   ROOT / 'tmp_work/training_free_fixed_check_20260930', POS / 'color']
    color_outputs = {str(base):json.loads((base / 'complete.json').read_text())['outputs'] for base in color_roots}
    input_paths = [V2 / 'manifest.json',V2 / 'complete.json',POS / 'manifest.json',POS / 'complete.json',
                   ROOT / 'tmp_work/preprocessing_status_before_warm_rescue_20260930.md']
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in input_paths})
    keys = ['red','robust_seed','protect','hold','edit_allowed','stain_candidate','dense_stain','dense_hold','protection_guard']
    for row in cases:
        hashes[row['original']] = sha(ROOT / row['original'])
        ref = ROOT / row['reference']
        for key in keys:
            p = ref / (key+'.png')
            expected = frozen_outputs[str(p.relative_to(V2))] if p.is_relative_to(V2) else positive_outputs[str(p.relative_to(POS))]
            assert sha(p) == expected
            hashes[str(p.relative_to(ROOT))] = expected
        p = ROOT / row['color'] / 'red_weak.png'
        color_base = next(base for base in color_roots if p.is_relative_to(base))
        expected = color_outputs[str(color_base)][str(p.relative_to(color_base))]
        assert sha(p) == expected
        hashes[str(p.relative_to(ROOT))] = expected
    manifest = dict(purpose='Development and regression; latest positive pages transferred to development',
                    source_sha256=sha(Path(__file__)),settings=SETTINGS,cases=cases,dependencies=hashes,
                    neural_training=False,pixel_GT=False,independent_test=False,
                    parameter_selection='First trial heuristics, not GT-optimized or calibrated')
    mp = OUT / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError('Existing version has changed code/settings/inputs; use a new directory')
    dump(mp,manifest);shutil.copyfile(__file__,OUT / 'source_snapshot.py')
    modes = dict(original='原画像',v2_edit='v2：編集許可',weak_edit='弱い赤条件のみ：編集許可',
                 global_edit='局所色・文脈なし：編集許可',context_edit='朱へのつながりあり：編集許可',
                 context_extra='朱へのつながりあり：追加保留',v2_protect='v2：文字保護（全方式同じ）')
    colors = dict(edit=(23,178,235),hold=(176,65,155),protect=(225,100,38))

    def overlay(original,mask,color):
        v = original.copy();v[mask] = np.round(.45*original[mask]+.55*np.array(color)).astype(np.uint8)
        return v

    metrics = [];cards = [];gallery = [];features = []
    for row in cases:
        started = time.monotonic();stem = row['id'];dest = OUT / stem
        original = read(ROOT / row['original']);h,w = original.shape[:2]
        ref = {key:read(ROOT / row['reference'] / (key+'.png'),True) for key in keys}
        weak = read(ROOT / row['color'] / 'red_weak.png',True)
        extras, feat = hold_candidates(original,ref,weak);feat.update(page=stem,group=row['group']);features.append(feat)
        for key,mask in extras.items():
            write(dest / (key+'.png'),mask)
        images = dict(original=original,v2_edit=overlay(original,ref['edit_allowed'],colors['edit']),
                      v2_protect=overlay(original,ref['protect'],colors['protect']))
        masks_by_mode = {}
        for mode,short in [('weak_red_hold','weak'),('global_warm_hold','global'),('context_warm_hold','context')]:
            extra = extras[mode];hold = ref['hold'] | extra;edit = ref['edit_allowed'] & ~extra
            assert not (edit & hold).any() and not (edit & ref['protect']).any()
            write(dest / (mode+'_hold.png'),hold);write(dest / (mode+'_edit_allowed.png'),edit)
            images[short+'_edit'] = overlay(original,edit,colors['edit'])
            masks_by_mode[mode] = dict(extra=extra,hold=hold,edit=edit)
        images['context_extra'] = overlay(original,extras['context_warm_hold'] & ~ref['hold'],colors['hold'])
        for key in ['protect','red','stain_candidate']:
            write(dest / (key+'.png'),ref[key])
        write(dest / 'original.png',original)
        dump(dest / 'features.json',feat)
        parts = [];page_gallery = dict(id=stem,group=row['group'],rois=[])
        rects = [(None,'ページ全体',[0,0,w,h])] + [(i,r['title'],r['xywh']) for i,r in enumerate(row['rois'],1)]
        for index,title,box in rects:
            x,y,rw,rh = box;sl = (slice(y,y+rh),slice(x,x+rw))
            entry = dict(page=stem,book=row['book'],group=row['group'],roi=index,title=title,xywh=box,
                         v2_edit_pixels=int(ref['edit_allowed'][sl].sum()),v2_protect_pixels=int(ref['protect'][sl].sum()))
            for mode,m in masks_by_mode.items():
                entry[mode+'_new_hold_pixels'] = int((m['extra'][sl] & ~ref['hold'][sl]).sum())
                entry[mode+'_edit_pixels'] = int(m['edit'][sl].sum())
                entry[mode+'_withheld_edit_pixels'] = int((ref['edit_allowed'][sl] & m['extra'][sl]).sum())
            metrics.append(entry)
            if index is None:
                continue
            panels = [];paths = {}
            for mode in modes:
                rel = f'{stem}/review/roi_{index:02d}_{mode}.png';write(OUT / rel,images[mode][sl]);paths[mode] = rel
                panels.append(f'<figure data-mode="{mode}"><figcaption>{html.escape(modes[mode])}</figcaption>'
                              f'<a href="{rel}"><img loading="lazy" src="{rel}" alt="{html.escape(modes[mode])}"></a></figure>')
            plot_modes = ['original','v2_edit','weak_edit','global_edit','context_edit','context_extra']
            fig,axes = plt.subplots(2,3,figsize=(13.5,2*max(2.,min(6.,4.5*rh/rw+.65))))
            for ax,mode in zip(axes.flat,plot_modes):
                ax.imshow(cv2.cvtColor(images[mode][sl],cv2.COLOR_BGR2RGB),interpolation='nearest')
                ax.set_title(modes[mode],fontproperties=font,fontsize=10);ax.axis('off')
            fig.suptitle(stem+' · '+title+'（開発資料・未注釈）',fontproperties=font,fontsize=12)
            fig.tight_layout(rect=(0,0,1,.95))
            rel = f'{stem}/review/roi_{index:02d}_comparison.png';fig.savefig(OUT / rel,dpi=145,facecolor='white');plt.close(fig)
            parts.append(f'<section><h3>{html.escape(title)} · {box}</h3><p>v2編集許可 {entry["v2_edit_pixels"]:,} → '
                         f'文脈付き保留後 {entry["context_warm_hold_edit_pixels"]:,}画素。減少は編集見送りの数で、精度ではない。</p>'
                         f'<p><a href="{rel}">6表示の比較図</a></p><div class="grid">{"".join(panels)}</div></section>')
            page_gallery['rois'].append(dict(title=title,xywh=box,comparison=rel,images=paths))
        links = ' · '.join(f'<a href="{stem}/{key}.png">{label}</a>' for key,label in [
            ('context_warm_hold','追加保留'),('context_warm_hold_edit_allowed','編集許可'),('protect','元の保護'),('red','元の朱候補')])
        cards.append(f'<article id="{stem}"><h2>{stem}</h2><p>冊 {row["book"]} · {row["group"]} · 原画像 {w}×{h}</p>'
                     f'<p>局所色に使える朱の種 {feat["anchor_pixels"]:,}画素 · 文脈の計算 {"あり" if feat["context_enabled"] else "なし"} · 追加保留対象の成分 {feat["accepted_components"]}</p>'
                     f'<details><summary>原寸マスク</summary><p>{links}</p></details>{"".join(parts)}</article>')
        gallery.append(page_gallery)
        full = next(m for m in reversed(metrics) if m['page']==stem and m['roi'] is None)
        print(f'{stem}: anchors={feat["anchor_pixels"]}, edit {full["v2_edit_pixels"]}->{full["context_warm_hold_edit_pixels"]}, '
              f'new hold={full["context_warm_hold_new_hold_pixels"]}, {time.monotonic()-started:.1f}s',flush=True)
        dump(OUT / 'progress.json',dict(status='running',completed_pages=len(gallery),total_pages=len(cases)))
    dump(OUT / 'diagnostics.json',metrics);dump(OUT / 'features.json',features);dump(OUT / 'gallery.json',gallery)
    # Diagnose the same input-selected failure zooms used in the preceding comparison.
    zooms = []
    for row in json.loads((POS / 'failure_zoom_locations.json').read_text()):
        stem = row['page'];x,y,rw,rh = row['xywh'];sl = (slice(y,y+rh),slice(x,x+rw))
        old_edit = read(POS / 'refined' / stem / 'edit_allowed.png',True)[sl]
        values = dict(original=read(OUT / stem / 'original.png')[sl],
                      old_edit=old_edit,new_edit=read(OUT / stem / 'context_warm_hold_edit_allowed.png',True)[sl],
                      extra_hold=read(OUT / stem / 'context_warm_hold.png',True)[sl])
        key = len(zooms)+1;panels = []
        for name in ['original','old_edit','new_edit','extra_hold']:
            image = values['original'] if name=='original' else overlay(values['original'],values[name],colors['hold' if name=='extra_hold' else 'edit'])
            path = OUT / 'failure_zooms' / f'zoom_{key:02d}_{name}.png';write(path,image)
            panel = np.full((rh+34,rw,3),255,np.uint8);panel[34:]=image
            cv2.putText(panel,name,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1);panels.append(panel)
        path = OUT / 'failure_zooms' / f'zoom_{key:02d}_comparison.png';write(path,np.concatenate(panels,axis=1))
        zooms.append(dict(page=stem,xywh=row['xywh'],image=str(path.relative_to(OUT)),
                          v2_edit_pixels=int(old_edit.sum()),context_edit_pixels=int(values['new_edit'].sum()),pixel_GT=False))
    dump(OUT / 'failure_zoom_diagnostics.json',zooms)
    controls = ''.join(f'<label><input type="checkbox" data-toggle="{key}" {"checked" if key in ["original","v2_edit","context_edit","context_extra"] else ""}>{html.escape(label)}</label>' for key,label in modes.items())
    toc = ' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in cases)
    (OUT / 'viewer.js').write_text('''const toggles=[...document.querySelectorAll('[data-toggle]')];function update(){for(const cb of toggles){for(const fig of document.querySelectorAll('figure[data-mode="'+cb.dataset.toggle+'"]'))fig.hidden=!cb.checked;}}toggles.forEach(cb=>cb.addEventListener('change',update));document.querySelector('#native').addEventListener('change',e=>document.body.classList.toggle('native',e.target.checked));update();\n''')
    viewer = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>薄い赤褐色の編集前保留：開発比較</title>
<style>body{font-family:system-ui,sans-serif;max-width:1800px;margin:24px auto;padding:0 18px;background:#f3f1ed;color:#202326;line-height:1.7}header,article,.controls{background:white;padding:20px;border-radius:10px;margin-bottom:22px}h1{font-size:1.6rem}h2{font-size:1.3rem}h3{font-size:1.1rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}figure{margin:0}figure img{width:100%;height:auto}figcaption{margin:8px 0}figure[hidden]{display:none}section{border-top:1px solid #ddd;padding:16px 0}.controls label{display:inline-block;margin:6px 18px 6px 0}a{color:#075ea0}nav{overflow-wrap:anywhere}body.native .grid{display:flex;flex-wrap:nowrap;overflow:auto;align-items:flex-start}body.native figure{flex:none}body.native figure img{width:auto;max-width:none}</style>
<header><h1>薄い赤褐色の編集前保留：開発比較</h1><p>21ページ・20冊・52か所。別冊9ページを開発資料へ移し、学習なし・画素GTなしで保留を試作した。朱の確定・文字保護・汚れ候補はv2のまま。編集許可から追加保留を引いた。黄＝編集許可、紫＝追加保留、青＝既存の保護。</p><p>局所的な明るい色との差・朱の種との色方向・成分の近接を用いる。薄い筆画の編集見送りと、汚れを保留してしまう副作用の両方を確認する。編集面積の減少は精度の向上ではなく、保留は文字と確定した判断ではない。</p><p><a href="report.md">所見と結果</a> · <a href="diagnostics.json">候補数</a> · <a href="verification.json">検証</a> · <a href="../preprocessing_status_before_warm_rescue_20260930.md">着手前の現状整理</a></p><nav>'''
    (OUT / 'index.html').write_text(viewer+toc+'</nav></header><div class="controls">'+controls+
        '<p><label><input type="checkbox" id="native">原寸表示（横スクロール）</label></p></div>'+''.join(cards)+'<script src="viewer.js"></script></html>\n')
    full = [m for m in metrics if m['roi'] is None]
    table = '\n'.join(f'| {m["page"]} | {m["v2_edit_pixels"]:,} | {m["weak_red_hold_edit_pixels"]:,} | {m["global_warm_hold_edit_pixels"]:,} | {m["context_warm_hold_edit_pixels"]:,} | {m["context_warm_hold_new_hold_pixels"]:,} |' for m in full)
    zoom_table = '\n'.join(f'| [{z["page"]}・拡大{i}]({z["image"]}) | {z["v2_edit_pixels"]:,} | {z["context_edit_pixels"]:,} |' for i,z in enumerate(zooms,1))
    (OUT / 'report.md').write_text(f'''# 薄い赤褐色の編集前保留：開発実験

## 現状からの変更

[着手前の現状整理](../preprocessing_status_before_warm_rescue_20260930.md)を保存してから、薄い朱印・赤褐色筆記の編集許可への混入を対象に保留を追加した。21ページ・20冊・52矩形は全て開発／回帰資料。別冊9ページの前回の固定比較を保存し、今回を独立評価とは呼ばない。学習・画素GT・画像削除・拡散復元は行っていない。

朱の確定マスク、文字保護、汚れ候補、保護ガードはv2のまま。今回は追加保留により編集許可だけを絞った。元画像の色や、保留に入った筆画の文字クラスを確定した実験ではない。

## 試作条件と比較

原BGRの3×3中央値画像から、4分の1の各チャンネルの31×31 closing・Gaussian sigma2で局所的に明るい色を推定し、原寸へ戻す。観測チャンネル以上に制限した値と入力の対数比を色差の手掛かりにする。物理的な紙色・顔料測定ではない。

赤チャンネルに対する緑・青の対数比差がともに0.012以上の部分を暖色候補とする。既存の確定朱と頑健な種の共通部分で両色差0.03以上の画素が20以上ある場合、その色方向の中央値±25度を使う。中央値は入力の統計量で、学習済みモデルを使っていない。

5px closingは連結関係の確認だけに使い、追加保留は元の色条件に入る画素だけ。成分は面積12〜100,000、矩形充填率0.75以下、縦横比35以下。既知の濃い汚れ・保留の成分画素を除く。朱の種からの成分最短距離512px以内で、最大64pxの近接を最大6回たどる。これは最初の試作条件で、最適値や不確実性の確率校正ではない。

比較は①保存済み弱赤条件だけの保留、②局所暖色・形だけの保留、③朱の色方向と近接を加えた保留。②は文脈の制約を外した診断用比較であり、採用方式ではない。実際に黒文字や茶色汚れと区別できるかは目視で確認する。

## ページ全体の編集候補数

| ページ | v2編集許可 | 弱赤のみ後 | 局所暖色・文脈なし後 | 朱の文脈あり後 | 文脈ありの追加保留 |
|---|---:|---:|---:|---:|---:|
{table}

追加保留には汚れが含まれ得る。編集を見送る画素の数を文字RecallやAccuracyと呼ばない。pixel数を独立標本数として使わず、21ページを20冊として扱う。

## 前回と同じ失敗箇所

| 拡大位置 | v2編集許可 | 文脈付き保留後の編集許可 |
|---|---:|---:|
{zoom_table}

これらは固定14矩形内の診断用拡大で、画素GTではない。画像の所見は `visual_findings.md` に記録する。

## 検証と限界

原画像・依存する既存マスク・コード・出力のハッシュ、原画素保存、各マスクの原寸・二値、朱・保護・汚れ候補の不変、編集許可がv2の部分集合で保護・保留・ガードと交差しないことを検証する。赤のない回帰8ページは文脈付き保留の追加が0で、編集許可も不変であることを確認する。既知の楕円状斑点3,131画素も編集候補に残るか確認する。これは代表的な精度評価ではない。

朱が近い茶色汚れ・暖色の墨や、色方向が似る撮影背景は誤保留の可能性がある。確定朱の種がないページは文脈の方式では救済できない。色が異なる赤や離れた孤立筆記、512pxや6回の制限外の筆画も取りこぼし得る。保留した画素の正しさを保証しない。ブラウザ実表示は未検証で、参照ファイルとJavaScript構文を検証する。
''')
    dump(OUT / 'summary.json',dict(development_pages=21,books=20,rois=52,
        v2_edit_pixels=sum(m['v2_edit_pixels'] for m in full),
        context_edit_pixels=sum(m['context_warm_hold_edit_pixels'] for m in full),
        context_new_hold_pixels=sum(m['context_warm_hold_new_hold_pixels'] for m in full)))
    dump(OUT / 'progress.json',dict(status='computed',completed_pages=21,total_pages=21))
    dump(OUT / 'verification.json',dict(status='pending'))
    products = [p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts
                and p.name not in ['complete.json','verification.json','run.log','progress.json','visual_findings.md']]
    dump(OUT / 'complete.json',dict(status='complete',outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(products)}))
    verify()
    dump(OUT / 'progress.json',dict(status='verified',completed_pages=21,total_pages=21))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else main()
