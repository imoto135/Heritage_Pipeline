"""Training-free warm hold v5: short red context and local ridge evidence.

Frozen v2/v3/v4 inputs; known development pages, no pixel GT or restoration.
Candidate area is a diagnostic, never an accuracy or recall score.
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

from refine_warm_ink_hold import ROOT, V2, V3, OUT as V4, KEYS, get_cases, sha, dump, read, write, distance_to, overlay
import cv2
import numpy as np

OUT = ROOT / 'tmp_work/warm_ink_hold_v5_20260930'
os.environ['MPLCONFIGDIR'] = str(OUT / 'plot_cache')
SETTINGS = dict(
    chromatic='Original median3, quarter-resolution closing31 ellipse + Gaussian2 bright field, original-pixel max; OD channel differences as v3',
    direction_tolerance_degrees=18., v3_page_anchor_gate=True,
    strength='min(OD_G-OD_R, OD_B-OD_R), before thresholding',
    ridge=dict(sigmas=[2., 4., 8.], derivative='Gaussian then Sobel second derivatives k3, scale sigma^2/4',
               eigen_rule='largest-absolute eigenvalue negative; |small| <= 0.5 |large|',
               response='-large * exp(-2*(small/(abs(large)+1e-8))^2); maximum across scales'),
    short_context=dict(max_red_distance=96., minimum_ridge=.005),
    supported_context=dict(support='Saved v4 red_protect_context_extra_hold', minimum_ridge=.025),
    neighbourhood=dict(dilation_radius=8, kernel='17px ellipse; only original chromatic pixels already editable become hold'),
    modes=['short_context: tighter color direction and red distance only',
           'stroke_context: short context with weak ridge, or v4 support with strong ridge'],
    fixed_decisions='Keep v2 confirmed red, protection, stain candidates and guard; extra hold subset of v2 edit',
    parameter_selection='Development heuristic trial after looking at 3 failure zooms and 2 brown diagnostic rectangles; no GT calibration',
    limitation='Ridges can also be stains; thresholds and distances in native pixels; not a physical pigment or ink classifier')
MODES = ['short_context', 'stroke_context']
NAMES = dict(original='原画像', v2_edit='保護v2：編集許可', v4_edit='暖色保留v4：編集許可',
             short_edit='短距離の朱文脈だけ：編集許可', stroke_edit='局所の線状変化も利用：編集許可',
             stroke_extra='今回の線状変化による追加保留')
COLORS = dict(edit=(23,178,235), hold=(176,65,155))


def fields(original):
    h,w = original.shape[:2]
    median = cv2.medianBlur(original,3)
    small = cv2.resize(median,((w+3)//4,(h+3)//4),interpolation=cv2.INTER_AREA)
    closed = cv2.morphologyEx(small,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)),borderType=cv2.BORDER_REFLECT)
    bright = cv2.GaussianBlur(closed.astype(np.float32),(0,0),2.,borderType=cv2.BORDER_REFLECT)
    bright = cv2.resize(bright,(w,h),interpolation=cv2.INTER_LINEAR)
    bright = np.maximum(bright,median.astype(np.float32))
    od = np.log((bright+1.)/(median.astype(np.float32)+1.))
    green = od[:,:,1]-od[:,:,2]; blue = od[:,:,0]-od[:,:,2]
    angle = np.arctan2(blue,green)
    strength = np.minimum(green,blue)
    return green,blue,angle,strength


def ridge_response(strength):
    response = np.zeros(strength.shape,np.float32)
    for sigma in SETTINGS['ridge']['sigmas']:
        smooth = cv2.GaussianBlur(strength,(0,0),sigma,borderType=cv2.BORDER_REFLECT)
        derivatives = [cv2.Sobel(smooth,cv2.CV_32F,dx,dy,ksize=3,scale=sigma*sigma*.25,
                                borderType=cv2.BORDER_REFLECT) for dx,dy in [(2,0),(0,2),(1,1)]]
        xx,yy,xy = derivatives
        root = np.sqrt((xx-yy)**2+4*xy**2)
        one = .5*(xx+yy-root); two = .5*(xx+yy+root)
        first = np.abs(one)>=np.abs(two)
        large = np.where(first,one,two); little = np.where(first,two,one)
        score = np.maximum(-large,0)*np.exp(-2*(little/(np.abs(large)+1e-8))**2)
        score[(large>=0)|(np.abs(little)>.5*np.abs(large))] = 0
        response = np.maximum(response,score)
    return response


def make_masks(original,ref,old_color,feat,v4_extra):
    green,blue,angle,strength = fields(original)
    warm = (green>=.012)&(blue>=.012)&~(ref['dense_stain']|ref['dense_hold'])
    center = feat['median_anchor_angle_degrees']
    base = warm if center is None else warm&(np.abs(angle-np.deg2rad(center))<=np.deg2rad(25.))
    assert np.array_equal(base,old_color), 'Recomputed v3 contextual color differs'
    enabled = feat['context_enabled']
    color = np.zeros(warm.shape,bool) if not enabled else warm&(np.abs(angle-np.deg2rad(center))<=np.deg2rad(18.))
    if not enabled:
        empty = np.zeros(warm.shape,bool)
        return dict(color_direction=color,ridge_weak_seed=empty,ridge_strong_seed=empty,
                    near_red=empty,short_context=empty,stroke_context=empty),dict(context_enabled=False,color_parity=True)
    ridge = ridge_response(strength)
    weak = color&(ridge>=.005); strong = color&(ridge>=.025)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    weak_neighbour = cv2.dilate(weak.astype(np.uint8),kernel)>0
    strong_neighbour = cv2.dilate(strong.astype(np.uint8),kernel)>0
    near = distance_to(ref['red'])<=96.
    editable_color = color&ref['edit_allowed']
    short = editable_color&near
    stroke = editable_color&((near&weak_neighbour)|(v4_extra&strong_neighbour))
    masks = dict(color_direction=color,ridge_weak_seed=weak,ridge_strong_seed=strong,
                 near_red=near,short_context=short,stroke_context=stroke)
    stats = dict(context_enabled=True,color_parity=True,weak_seed_pixels=int(weak.sum()),
                 strong_seed_pixels=int(strong.sum()),short_distance_pixels=int((near&color).sum()),
                 short_ridge_added_hold=int((editable_color&near&weak_neighbour).sum()),
                 supported_ridge_added_hold=int((editable_color&v4_extra&strong_neighbour).sum()))
    return masks,stats


def verify():
    m = json.loads((OUT/'manifest.json').read_text())
    assert m['source_sha256']==sha(Path(__file__)) and m['settings']==SETTINGS
    for p,h in m['dependencies'].items(): assert sha(ROOT/p)==h,p
    c = json.loads((OUT/'complete.json').read_text())
    for p,h in c['outputs'].items(): assert sha(OUT/p)==h,p
    negative=[];parity=[]
    for row in m['cases']:
        dest=OUT/row['id']; ref={k:read(ROOT/row['reference']/(k+'.png'),True) for k in KEYS}
        original=read(ROOT/row['original'])
        assert np.array_equal(original,read(dest/'original.png'))
        for k in ['red','protect','stain_candidate']: assert np.array_equal(ref[k],read(dest/(k+'.png'),True))
        assert json.loads((dest/'features.json').read_text())['color_parity'];parity.append(row['id'])
        for mode in MODES:
            extra=read(dest/(mode+'_extra_hold.png'),True); held=read(dest/(mode+'_hold.png'),True);edit=read(dest/(mode+'_edit_allowed.png'),True)
            assert not (extra&~ref['edit_allowed']).any()
            assert np.array_equal(held,ref['hold']|extra) and np.array_equal(edit,ref['edit_allowed']&~extra)
            assert not (edit&~ref['stain_candidate']).any()
            for other in [held,ref['protect'],ref['red'],ref['protection_guard']]: assert not (edit&other).any()
            if row['group']=='negative_regression': assert not extra.any() and np.array_equal(edit,ref['edit_allowed'])
        if row['group']=='negative_regression':negative.append(row['id'])
        for p in dest.glob('*.png'):
            if p.name=='original.png':continue
            v=cv2.imread(str(p),0); assert v.shape==original.shape[:2] and set(np.unique(v)).issubset({0,255})
        maps=dict(v2_edit=ref['edit_allowed'],v4_edit=read(V4/row['id']/'red_protect_context_edit_allowed.png',True),
                  short_edit=read(dest/'short_context_edit_allowed.png',True),stroke_edit=read(dest/'stroke_context_edit_allowed.png',True),
                  stroke_extra=read(dest/'stroke_context_extra_hold.png',True))
        for i,roi in enumerate(row['rois'],1):
            x,y,w,h=roi['xywh'];sl=(slice(y,y+h),slice(x,x+w))
            assert np.array_equal(read(dest/'review'/f'roi_{i:02d}_original.png'),original[sl])
            for key,mask in maps.items():
                assert np.array_equal(read(dest/'review'/f'roi_{i:02d}_{key}.png'),overlay(original[sl],mask[sl],COLORS['hold' if key=='stroke_extra' else 'edit']))
    sl=(slice(591,646),slice(461,545))
    dense=read(V2/'hnsd001_011/dense_stain.png',True)[sl];edit=read(OUT/'hnsd001_011/stroke_context_edit_allowed.png',True)[sl]
    assert int(dense.sum())==3131 and int((dense&edit).sum())==3131
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and v and not v.startswith('#'):assert (OUT/v).is_file(),v
    Links().feed((OUT/'index.html').read_text())
    subprocess.run(['node','--check',str(OUT/'viewer.js')],check=True,cwd=ROOT)
    result=dict(status='passed',pages=21,books=20,fixed_rois=52,output_hashes_checked=len(c['outputs']),
                inputs_code_and_outputs_hashed=True,original_pixels_and_native_overlays_checked=True,
                native_binary_masks=True,red_protect_stain_unchanged=True,extra_hold_subset_of_v2_edit=True,
                edit_excludes_hold_red_protect_guard=True,color_recomputation_parity_pages=parity,
                negative_pages_unchanged=negative,known_dense_stain_still_editable_pixels=3131,
                browser_rendering_tested=False,viewer_links_and_javascript_checked=True,
                pixel_GT=False,neural_training=False,independent_accuracy_test=False)
    dump(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    cases=get_cases(); prior=json.loads((V4/'manifest.json').read_text());assert cases==prior['cases']
    dependencies=dict(prior['dependencies']);products=json.loads((V4/'complete.json').read_text())['outputs']
    for p in [V4/'manifest.json',V4/'complete.json',Path(__file__).with_name('refine_warm_ink_hold.py'),
              OUT/'initial_feature_diagnostics.json']:
        dependencies[str(p.relative_to(ROOT))]=sha(p)
    for row in cases:
        for name in ['red_protect_context_extra_hold.png','red_protect_context_edit_allowed.png']:
            p=V4/row['id']/name;h=products[str(p.relative_to(V4))]; assert sha(p)==h
            dependencies[str(p.relative_to(ROOT))]=h
    for p,h in dependencies.items():assert sha(ROOT/p)==h,p
    manifest=dict(source_sha256=sha(Path(__file__)),settings=SETTINGS,cases=cases,dependencies=dependencies,
                  purpose='Known development/regression only; no accuracy claim',pixel_GT=False,neural_training=False)
    mp=OUT/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Changed experiment: use a new version directory')
    dump(mp,manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    metrics=[];cards=[]
    for row in cases:
        start=time.monotonic();stem=row['id'];dest=OUT/stem
        original=read(ROOT/row['original']);h,w=original.shape[:2]
        ref={k:read(ROOT/row['reference']/(k+'.png'),True) for k in KEYS}
        color=read(V3/stem/'contextual_candidate.png',True);feat=json.loads((V3/stem/'features.json').read_text())
        v4_extra=read(V4/stem/'red_protect_context_extra_hold.png',True)
        masks,stats=make_masks(original,ref,color,feat,v4_extra)
        for key,v in masks.items():write(dest/(key+'.png'),v)
        write(dest/'original.png',original)
        for key in ['red','protect','stain_candidate']:write(dest/(key+'.png'),ref[key])
        dump(dest/'features.json',stats)
        edits={}
        for mode in MODES:
            extra=masks[mode];edits[mode]=ref['edit_allowed']&~extra
            write(dest/(mode+'_extra_hold.png'),extra);write(dest/(mode+'_hold.png'),ref['hold']|extra);write(dest/(mode+'_edit_allowed.png'),edits[mode])
        v4_edit=read(V4/stem/'red_protect_context_edit_allowed.png',True)
        images=dict(original=original,v2_edit=overlay(original,ref['edit_allowed'],COLORS['edit']),
                    v4_edit=overlay(original,v4_edit,COLORS['edit']),short_edit=overlay(original,edits['short_context'],COLORS['edit']),
                    stroke_edit=overlay(original,edits['stroke_context'],COLORS['edit']),stroke_extra=overlay(original,masks['stroke_context'],COLORS['hold']))
        parts=[]
        rects=[(None,'ページ全体',[0,0,w,h])]+[(i,r['title'],r['xywh']) for i,r in enumerate(row['rois'],1)]
        for index,title,box in rects:
            x,y,rw,rh=box;sl=(slice(y,y+rh),slice(x,x+rw))
            entry=dict(page=stem,book=row['book'],group=row['group'],roi=index,title=title,xywh=box,
                       v2_edit_pixels=int(ref['edit_allowed'][sl].sum()),v4_edit_pixels=int(v4_edit[sl].sum()),
                       **{mode+'_edit_pixels':int(edit[sl].sum()) for mode,edit in edits.items()})
            metrics.append(entry)
            if index is None:continue
            panels=[]
            for key,label in NAMES.items():
                rel=f'{stem}/review/roi_{index:02d}_{key}.png';write(OUT/rel,images[key][sl])
                panels.append(f'<figure data-mode="{key}"><figcaption>{html.escape(label)}</figcaption><a href="{rel}"><img loading="lazy" src="{rel}" alt="{html.escape(label)}"></a></figure>')
            parts.append(f'<section><h3>{html.escape(title)} · {box}</h3><p>v2編集候補 {entry["v2_edit_pixels"]:,} → v4 {entry["v4_edit_pixels"]:,} → 今回 {entry["stroke_context_edit_pixels"]:,}画素。候補数であり精度ではない。</p><div class="grid">{"".join(panels)}</div></section>')
        cards.append(f'<article id="{stem}"><h2>{stem}</h2><p>冊 {row["book"]} · {row["group"]} · 原寸{w}×{h}</p>{"".join(parts)}</article>')
        print(f'{stem}: v2 edit={int(ref["edit_allowed"].sum())}, v4={int(v4_edit.sum())}, short={int(edits["short_context"].sum())}, stroke={int(edits["stroke_context"].sum())}, {time.monotonic()-start:.1f}s',flush=True)
        dump(OUT/'progress.json',dict(status='running',completed_pages=len(cards),total_pages=21))
    dump(OUT/'diagnostics.json',metrics)
    zooms=json.loads((ROOT/'tmp_work/red_positive_fixed_check_20260930/failure_zoom_locations.json').read_text())
    for i,z in enumerate(zooms,1):z['title']=['長い赤褐色の払い','上側の赤褐色筆記','薄い朱印'][i-1]
    zooms += [dict(page='200010454_00002_2',xywh=[1655,1810,75,195],title='朱印近傍の茶色・折り目'),
              dict(page='200010454_00002_2',xywh=[710,200,450,330],title='離れた茶色い汚れ')]
    zoom_results=[]
    for i,z in enumerate(zooms,1):
        stem=z['page'];x,y,w,h=z['xywh'];sl=(slice(y,y+h),slice(x,x+w));r=next(r for r in cases if r['id']==stem)
        original=read(OUT/stem/'original.png')[sl]
        masks=dict(v2_edit=read(ROOT/r['reference']/'edit_allowed.png',True)[sl],v4_edit=read(V4/stem/'red_protect_context_edit_allowed.png',True)[sl],
                   short_edit=read(OUT/stem/'short_context_edit_allowed.png',True)[sl],stroke_edit=read(OUT/stem/'stroke_context_edit_allowed.png',True)[sl],
                   stroke_extra=read(OUT/stem/'stroke_context_extra_hold.png',True)[sl])
        fig,axes=plt.subplots(2,3,figsize=(13.5,2*max(2.,min(6.,4.5*h/w+.65))))
        for ax,(key,label) in zip(axes.flat,NAMES.items()):
            picture=original if key=='original' else overlay(original,masks[key],COLORS['hold' if key=='stroke_extra' else 'edit'])
            write(OUT/'failure_zooms'/f'zoom_{i:02d}_{key}.png',picture)
            ax.imshow(cv2.cvtColor(picture,cv2.COLOR_BGR2RGB),interpolation='nearest');ax.set_title(label,fontproperties=font,fontsize=10);ax.axis('off')
        fig.suptitle(z['title']+'（診断矩形・画素GTなし）',fontproperties=font);fig.tight_layout(rect=(0,0,1,.95))
        rel=f'failure_zooms/zoom_{i:02d}_comparison.png';fig.savefig(OUT/rel,dpi=145,facecolor='white');plt.close(fig)
        zoom_results.append(dict(page=stem,title=z['title'],xywh=z['xywh'],image=rel,pixel_GT=False,
                                 **{k+'_pixels':int(v.sum()) for k,v in masks.items()}))
    dump(OUT/'failure_zoom_diagnostics.json',zoom_results)
    full=[m for m in metrics if m['roi'] is None]
    dump(OUT/'summary.json',dict(pages=21,books=20,rois=52,**{key:sum(m[key] for m in full) for key in ['v2_edit_pixels','v4_edit_pixels','short_context_edit_pixels','stroke_context_edit_pixels']}))
    table='\n'.join(f'| [{z["title"]}]({z["image"]}) | {z["v2_edit_pixels"]:,} | {z["v4_edit_pixels"]:,} | {z["short_edit_pixels"]:,} | {z["stroke_edit_pixels"]:,} |' for z in zoom_results)
    (OUT/'report.md').write_text(f'''# 暖色の局所線状変化を使う保留試作v5

前回v4の薄い朱印の取りこぼしと、朱近傍の茶色汚れの誤保留を対象に、学習なしで比較した。21ページ・20冊・52矩形は既知の開発／回帰資料。3失敗拡大と2茶色診断矩形を先に調べて条件を決めたため、独立評価ではない。画素GT、ニューラル学習、事前学習モデル推論、画像削除、拡散復元は行っていない。

## 条件

局所の明るい色との対数比差はv3と同じ計算で、保存済み色候補との全ページ画素一致を確認する。色の方向を朱の種の中央値±25度から±18度へ絞る。これは入力の色統計で、確定インクの認定や物理的な顔料測定ではない。朱の種が20未満のページは追加保留なし。

色の強さ `min(OD_G-OD_R,OD_B-OD_R)` のGaussian sigma2/4/8からSobel二階微分を計算する。最大絶対値の固有値が負で、小さい固有値の絶対値がその半分以下の場所に、線状変化の応答を持たせる。応答は `-large * exp(-2*(small/(abs(large)+1e-8))^2)` の尺度間最大。汚れも線状応答を持ち得るので、それだけで文字と断定しない。

比較①は色条件と確定朱から96px以内だけ。②は同じ近傍で応答0.005以上、またはv4で追加保留だった場所で応答0.025以上を手掛かりとする。応答の種から8pxの楕円近傍を使うが、追加保留は元の色条件に入る旧v2編集候補だけ。人工的な筆画を追加しない。条件は開発用の経験則で、しきい値の最適性や解像度をまたぐ校正を主張しない。

v2から保留を作り直すため、v4で誤保留した汚れを再び編集候補に戻す場合がある。v4に保留を足すだけの方式ではない。確定朱・文字保護・汚れ候補・保護ガードはv2のまま。

## 前回と同じ失敗位置と茶色対照

| 診断矩形 | v2編集候補 | v4 | ①短距離の朱文脈 | ②局所線状変化も利用 |
|---|---:|---:|---:|---:|
{table}

数字は矩形内の編集候補で、文字や汚れのGT画素数ではない。少なくするだけで方式を選ばず、文字の見送りと汚れの誤保留を両方確認する。目視所見と採用判断は `visual_findings.md` に別途記録する。

## 検証と限界

原画像・既存マスク・コード・出力のハッシュ、原寸二値マスク、52矩形の原寸画像とオーバーレイ、朱・保護・汚れ候補の不変、編集対象の包含・除外を検証する。赤のない8ページの不変と、既知の濃い楕円状斑点3,131画素の編集対象保持を確認する。ビューアのリンクとJavaScript構文を確認し、ブラウザ実表示は未検証。

太い・一様な筆画や色方向の異なる朱、既存支持のない孤立筆記、朱に近い細長い汚れは難例。元のCSC／ACE・原論文への総合的優越を示した実験ではない。今回も全資料を開発用として扱い、最終確認と評価用GTは後の段階に残す。
''')
    controls=''.join(f'<label><input type="checkbox" data-toggle="{key}" {"checked" if key in ["original","v4_edit","stroke_edit","stroke_extra"] else ""}>{html.escape(label)}</label>' for key,label in NAMES.items())
    toc=' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in cases)
    zoom_links=' · '.join(f'<a href="{z["image"]}">{html.escape(z["title"])}</a>' for z in zoom_results)
    (OUT/'viewer.js').write_text("const cs=[...document.querySelectorAll('[data-toggle]')];function update(){for(const c of cs)for(const f of document.querySelectorAll('figure[data-mode=\"'+c.dataset.toggle+'\"]'))f.hidden=!c.checked;}cs.forEach(c=>c.addEventListener('change',update));document.querySelector('#native').addEventListener('change',e=>document.body.classList.toggle('native',e.target.checked));update();\n")
    (OUT/'index.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>暖色保留v5：局所の線状変化</title>
<style>body{font-family:system-ui,sans-serif;max-width:1800px;margin:24px auto;padding:0 18px;background:#f3f1ed;color:#202326;line-height:1.7}header,article,.controls{background:white;padding:20px;border-radius:10px;margin-bottom:22px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}figure{margin:0}figure img{width:100%;height:auto}figure[hidden]{display:none}section{border-top:1px solid #ddd;padding:16px 0}label{display:inline-block;margin:6px 16px 6px 0}a{color:#075ea0}nav{overflow-wrap:anywhere}body.native .grid{display:flex;flex-wrap:nowrap;overflow:auto;align-items:flex-start}body.native figure{flex:none}body.native figure img{width:auto;max-width:none}</style>
<header><h1>暖色保留v5：局所の線状変化</h1><p>21ページ・20冊・52か所。既知の開発／回帰資料。黄＝編集許可、紫＝今回の追加保留。薄い朱の断片と朱近傍の汚れを比較する。保留は文字の認定ではなく編集見送り。候補数を精度と呼ばない。</p><p>色方向・短距離の朱文脈・局所の線状変化を使う。学習・手動画素GT・画像削除・拡散復元は実施していない。</p><p><a href="report.md">条件と結果</a> · <a href="verification.json">検証</a> · <a href="../warm_ink_hold_v4_20260930/visual_findings.md">前回の所見</a></p><p>'''+zoom_links+'</p><nav>'+toc+'</nav></header><div class="controls">'+controls+'<p><label><input type="checkbox" id="native">原寸表示（横スクロール）</label></p></div>'+''.join(cards)+'<script src="viewer.js"></script></html>\n')
    dump(OUT/'verification.json',dict(status='pending'))
    files=[p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts and p.name not in ['complete.json','verification.json','progress.json','run.log']]
    dump(OUT/'complete.json',dict(status='complete',outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(files)}))
    verify();dump(OUT/'progress.json',dict(status='verified',completed_pages=21,total_pages=21))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    verify() if args.verify_only else main()
