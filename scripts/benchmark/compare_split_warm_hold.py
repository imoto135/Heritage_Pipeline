"""Warm hold v6: change only output color gates, reuse frozen v5 ridge seeds."""
import argparse
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from compare_warm_stroke_hold import ROOT, V2, V3, V4, OUT as V5, read, write, dump, sha, overlay, COLORS
import cv2
import numpy as np

OUT=ROOT/'tmp_work/warm_ink_hold_v6_20260930'
ABLATION=ROOT/'tmp_work/warm_stroke_ablation_v5_20260930'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
SETTINGS=dict(
    experiment='Change output color gates only; all ridge seeds and support frozen from v5',
    short_path='v5 narrow18 color & v2 edit & v5 near_red96 & dilated v5 weak ridge seeds',
    supported_path='v3 broad25 color & v2 edit & v4 extra_hold & dilated v5 strong ridge seeds',
    split_direction='short_path | supported_path',
    wide_direction='v3 broad25 color & v2 edit & ((near_red96 & weak seed neighbours) | (v4 extra & strong seed neighbours))',
    seed_neighbourhood='17px ellipse dilation, same 8px radius as v5; seed color still narrow18',
    fixed='Original pixels, confirmed red, protection, stain candidate, guard, v4 support, v5 seeds unchanged',
    parameter_selection='Known development examples; separate output gates after matched ablation; no GT calibration',
    limitation='Diagnostic extra hold does not certify ink; native-pixel thresholds and inherited v4 mistakes')
MODES=['split_direction','wide_direction']
NAMES=dict(original='原画像',v4_edit='暖色保留v4：編集許可',v5_edit='暖色保留v5：編集許可',
           split_edit='経路ごとの色条件：編集許可',wide_edit='両経路の色条件を広げる：編集許可',
           regained_hold='v5から増えた保留')


def make_masks(old,narrow,broad,near,v4_extra,weak_seed,strong_seed,v5_extra):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
    weak=cv2.dilate(weak_seed.astype(np.uint8),kernel)>0
    strong=cv2.dilate(strong_seed.astype(np.uint8),kernel)>0
    control=old&narrow&((near&weak)|(v4_extra&strong))
    assert np.array_equal(control,v5_extra), 'Frozen v5 formula parity failed'
    short=old&narrow&near&weak
    supported=old&broad&v4_extra&strong
    split=short|supported
    wide=old&broad&((near&weak)|(v4_extra&strong))
    assert not (v5_extra&~split).any() and not (split&~wide).any()
    return dict(split_direction=split,wide_direction=wide,regained_hold=split&~v5_extra)


def verify():
    m=json.loads((OUT/'manifest.json').read_text());assert m['source_sha256']==sha(Path(__file__)) and m['settings']==SETTINGS
    for p,h in m['dependencies'].items():assert sha(ROOT/p)==h,p
    c=json.loads((OUT/'complete.json').read_text())
    for p,h in c['outputs'].items():assert sha(OUT/p)==h,p
    negative=[];parity=[]
    for row in m['cases']:
        stem=row['id'];dest=OUT/stem;original=read(ROOT/row['original'])
        assert np.array_equal(original,read(dest/'original.png'))
        old=read(ROOT/row['reference']/'edit_allowed.png',True)
        old_hold=read(ROOT/row['reference']/'hold.png',True)
        for k in ['red','protect','stain_candidate']:assert np.array_equal(read(dest/(k+'.png'),True),read(ROOT/row['reference']/(k+'.png'),True))
        v5_extra=read(V5/stem/'stroke_context_extra_hold.png',True)
        split=read(dest/'split_direction_extra_hold.png',True);wide=read(dest/'wide_direction_extra_hold.png',True)
        assert not (v5_extra&~split).any() and not (split&~wide).any()
        assert np.array_equal(read(dest/'regained_hold.png',True),split&~v5_extra)
        assert json.loads((dest/'features.json').read_text())['frozen_v5_parity'];parity.append(stem)
        for mode in MODES:
            extra=read(dest/(mode+'_extra_hold.png'),True);edit=read(dest/(mode+'_edit_allowed.png'),True);held=read(dest/(mode+'_hold.png'),True)
            assert not (extra&~old).any() and np.array_equal(edit,old&~extra) and np.array_equal(held,old_hold|extra)
            for k in ['red','protect','protection_guard']:assert not (edit&read(ROOT/row['reference']/(k+'.png'),True)).any()
            assert not (edit&held).any()
            if row['group']=='negative_regression':assert not extra.any() and np.array_equal(edit,old)
        if row['group']=='negative_regression':negative.append(stem)
        for p in dest.glob('*.png'):
            if p.name=='original.png':continue
            value=cv2.imread(str(p),0);assert value.shape==original.shape[:2] and set(np.unique(value)).issubset({0,255})
        maps=dict(v4_edit=read(V4/stem/'red_protect_context_edit_allowed.png',True),v5_edit=read(V5/stem/'stroke_context_edit_allowed.png',True),
                  split_edit=read(dest/'split_direction_edit_allowed.png',True),wide_edit=read(dest/'wide_direction_edit_allowed.png',True),regained_hold=split&~v5_extra)
        for i,r in enumerate(row['rois'],1):
            x,y,w,h=r['xywh'];sl=(slice(y,y+h),slice(x,x+w))
            assert np.array_equal(read(dest/'review'/f'roi_{i:02d}_original.png'),original[sl])
            for key,mask in maps.items():assert np.array_equal(read(dest/'review'/f'roi_{i:02d}_{key}.png'),overlay(original[sl],mask[sl],COLORS['hold' if key=='regained_hold' else 'edit']))
    sl=(slice(591,646),slice(461,545));dense=read(V2/'hnsd001_011/dense_stain.png',True)[sl];edit=read(OUT/'hnsd001_011/split_direction_edit_allowed.png',True)[sl]
    assert int(dense.sum())==3131 and int((dense&edit).sum())==3131
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and v and not v.startswith('#'):assert (OUT/v).is_file(),v
    Links().feed((OUT/'index.html').read_text());subprocess.run(['node','--check',str(OUT/'viewer.js')],check=True,cwd=ROOT)
    result=dict(status='passed',pages=21,books=20,rois=52,output_hashes_checked=len(c['outputs']),
                input_code_output_hashes_checked=True,original_and_native_overlay_pixels_checked=True,
                native_binary_masks=True,red_protect_stain_unchanged=True,extra_subset_of_v2_edit=True,
                v5_hold_subset_of_split_subset_of_wide=True,v5_formula_parity_pages=parity,
                negative_pages_unchanged=negative,known_dense_stain_still_editable_pixels=3131,
                viewer_links_and_javascript_checked=True,browser_rendering_tested=False,
                pixel_GT=False,neural_training=False,independent_accuracy_test=False)
    dump(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    prior=json.loads((V5/'manifest.json').read_text());cases=prior['cases'];dependencies=dict(prior['dependencies'])
    products=json.loads((V5/'complete.json').read_text())['outputs']
    for p in [V5/'manifest.json',V5/'complete.json',Path(__file__).with_name('compare_warm_stroke_hold.py'),
              ABLATION/'manifest.json',ABLATION/'complete.json']:
        dependencies[str(p.relative_to(ROOT))]=sha(p)
    for row in cases:
        for name in ['original.png','red.png','protect.png','stain_candidate.png','color_direction.png','near_red.png',
                     'ridge_weak_seed.png','ridge_strong_seed.png','stroke_context_extra_hold.png','stroke_context_edit_allowed.png']:
            p=V5/row['id']/name;h=products[str(p.relative_to(V5))];assert sha(p)==h;dependencies[str(p.relative_to(ROOT))]=h
    for p,h in dependencies.items():assert sha(ROOT/p)==h,p
    manifest=dict(source_sha256=sha(Path(__file__)),settings=SETTINGS,cases=cases,dependencies=dependencies,purpose='Known development/regression; not independent accuracy',pixel_GT=False,neural_training=False)
    mp=OUT/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Use new version directory for changed inputs/code')
    dump(mp,manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    metrics=[];cards=[]
    for row in cases:
        start=time.monotonic();stem=row['id'];dest=OUT/stem;original=read(V5/stem/'original.png');h,w=original.shape[:2]
        old=read(ROOT/row['reference']/'edit_allowed.png',True);held=read(ROOT/row['reference']/'hold.png',True)
        narrow=read(V5/stem/'color_direction.png',True);broad=read(V3/stem/'contextual_candidate.png',True)
        near=read(V5/stem/'near_red.png',True);support=read(V4/stem/'red_protect_context_extra_hold.png',True)
        weak=read(V5/stem/'ridge_weak_seed.png',True);strong=read(V5/stem/'ridge_strong_seed.png',True)
        v5_extra=read(V5/stem/'stroke_context_extra_hold.png',True)
        masks=make_masks(old,narrow,broad,near,support,weak,strong,v5_extra)
        write(dest/'original.png',original);write(dest/'regained_hold.png',masks['regained_hold'])
        for k in ['red','protect','stain_candidate']:write(dest/(k+'.png'),read(V5/stem/(k+'.png'),True))
        dump(dest/'features.json',dict(frozen_v5_parity=True,ridge_seeds_recomputed=False,color_change_only_in_output=True,regained_hold_pixels=int(masks['regained_hold'].sum())))
        edits={}
        for mode in MODES:
            edits[mode]=old&~masks[mode]
            write(dest/(mode+'_extra_hold.png'),masks[mode]);write(dest/(mode+'_hold.png'),held|masks[mode]);write(dest/(mode+'_edit_allowed.png'),edits[mode])
        v4_edit=read(V4/stem/'red_protect_context_edit_allowed.png',True);v5_edit=read(V5/stem/'stroke_context_edit_allowed.png',True)
        images=dict(original=original,v4_edit=overlay(original,v4_edit,COLORS['edit']),v5_edit=overlay(original,v5_edit,COLORS['edit']),
                    split_edit=overlay(original,edits['split_direction'],COLORS['edit']),wide_edit=overlay(original,edits['wide_direction'],COLORS['edit']),regained_hold=overlay(original,masks['regained_hold'],COLORS['hold']))
        parts=[]
        rects=[(None,'ページ全体',[0,0,w,h])]+[(i,r['title'],r['xywh']) for i,r in enumerate(row['rois'],1)]
        for i,title,box in rects:
            x,y,rw,rh=box;sl=(slice(y,y+rh),slice(x,x+rw))
            entry=dict(page=stem,book=row['book'],group=row['group'],roi=i,title=title,xywh=box,v2_edit_pixels=int(old[sl].sum()),v4_edit_pixels=int(v4_edit[sl].sum()),v5_edit_pixels=int(v5_edit[sl].sum()),
                       split_edit_pixels=int(edits['split_direction'][sl].sum()),wide_edit_pixels=int(edits['wide_direction'][sl].sum()),regained_hold_pixels=int(masks['regained_hold'][sl].sum()))
            metrics.append(entry)
            if i is None:continue
            panels=[]
            for key,label in NAMES.items():
                rel=f'{stem}/review/roi_{i:02d}_{key}.png';write(OUT/rel,images[key][sl])
                panels.append(f'<figure data-mode="{key}"><figcaption>{html.escape(label)}</figcaption><a href="{rel}"><img src="{rel}" loading="lazy" alt="{html.escape(label)}"></a></figure>')
            parts.append(f'<section><h3>{html.escape(title)} · {box}</h3><p>v4 {entry["v4_edit_pixels"]:,} → v5 {entry["v5_edit_pixels"]:,} → 経路ごとの色条件 {entry["split_edit_pixels"]:,}画素。候補数は精度ではない。</p><div class="grid">{"".join(panels)}</div></section>')
        cards.append(f'<article id="{stem}"><h2>{stem}</h2><p>冊{row["book"]} · {row["group"]}</p>{"".join(parts)}</article>')
        print(f'{stem}: v5={int(v5_edit.sum())}, split={int(edits["split_direction"].sum())}, wide={int(edits["wide_direction"].sum())}, {time.monotonic()-start:.1f}s',flush=True)
        dump(OUT/'progress.json',dict(status='running',completed_pages=len(cards),total_pages=21))
    dump(OUT/'diagnostics.json',metrics)
    zooms=[]
    for i,z in enumerate(json.loads((V5/'failure_zoom_diagnostics.json').read_text()),1):
        stem=z['page'];x,y,w,h=z['xywh'];sl=(slice(y,y+h),slice(x,x+w));original=read(OUT/stem/'original.png')[sl]
        masks=dict(v4_edit=read(V4/stem/'red_protect_context_edit_allowed.png',True)[sl],v5_edit=read(V5/stem/'stroke_context_edit_allowed.png',True)[sl],
                   split_edit=read(OUT/stem/'split_direction_edit_allowed.png',True)[sl],wide_edit=read(OUT/stem/'wide_direction_edit_allowed.png',True)[sl],regained_hold=read(OUT/stem/'regained_hold.png',True)[sl])
        fig,axes=plt.subplots(2,3,figsize=(13.5,2*max(2.,min(6.,4.5*h/w+.65))))
        for ax,(key,label) in zip(axes.flat,NAMES.items()):
            picture=original if key=='original' else overlay(original,masks[key],COLORS['hold' if key=='regained_hold' else 'edit'])
            write(OUT/'failure_zooms'/f'zoom_{i:02d}_{key}.png',picture)
            ax.imshow(cv2.cvtColor(picture,cv2.COLOR_BGR2RGB),interpolation='nearest');ax.set_title(label,fontproperties=font,fontsize=10);ax.axis('off')
        fig.suptitle(z['title']+'（既知開発位置・GTなし）',fontproperties=font);fig.tight_layout(rect=(0,0,1,.95))
        rel=f'failure_zooms/zoom_{i:02d}_comparison.png';fig.savefig(OUT/rel,dpi=145,facecolor='white');plt.close(fig)
        zooms.append(dict(page=stem,title=z['title'],xywh=z['xywh'],image=rel,pixel_GT=False,**{k+'_pixels':int(v.sum()) for k,v in masks.items()}))
    dump(OUT/'failure_zoom_diagnostics.json',zooms)
    full=[m for m in metrics if m['roi'] is None];keys=['v2_edit_pixels','v4_edit_pixels','v5_edit_pixels','split_edit_pixels','wide_edit_pixels','regained_hold_pixels']
    dump(OUT/'summary.json',dict(pages=21,books=20,rois=52,**{key:sum(m[key] for m in full) for key in keys}))
    table='\n'.join(f'| [{z["title"]}]({z["image"]}) | {z["v4_edit_pixels"]:,} | {z["v5_edit_pixels"]:,} | {z["split_edit_pixels"]:,} | {z["wide_edit_pixels"]:,} |' for z in zooms)
    (OUT/'report.md').write_text(f'''# 暖色保留v6：経路ごとの出力色条件

ユーザーの「現在のものが終わったら分析して次に進んでおいて」に従い、[v5の分析](../warm_ink_hold_v5_20260930/visual_findings.md)と[線状条件だけを外す比較](../warm_stroke_ablation_v5_20260930/report.md)を保存してから次の修正を実行した。21ページ・20冊・52矩形は全て既知の開発／回帰資料。画素GT・ニューラル学習・事前学習モデル推論・画像削除・拡散復元は今回も行っていない。

## 分析からの変更

v5は薄い朱印の編集候補を減らし、既知の朱近傍の茶色対照を編集候補へ戻した。一方、長い払いの取りこぼしが増えた。同じ色・文脈で線状条件を外しても筆記の候補数は同じなので、その位置の悪化は出力の色方向を狭めた変更による。

v6は出力の色条件だけを経路で分ける。朱96px近傍と弱い線状応答による経路はv5の±18度を維持。v4の保留支持と強い線状応答による経路だけ、出力色を元のv3の±25度へ戻す。比較として両経路の出力色を±25度へ広げる方式も計算した。**線状応答の種はv5の±18度のものをそのまま使い、再計算・追加しない。** 近傍の17px楕円dilationも同じ。色ゲートだけの影響を確認するためである。

確定朱・文字保護・汚れ候補・保護ガードはv2のまま。追加保留は旧v2編集候補だけ。v5の保留は経路別色条件の保留の部分集合で、さらに全経路拡大の部分集合。追加する画素を文字と認定しない。

## 同じ失敗位置と茶色対照

| 同じ診断拡大の編集候補 | v4 | v5 | 経路ごとの出力色条件 | 両経路の出力色を広げる |
|---|---:|---:|---:|---:|
{table}

候補数はAccuracy／Recallではない。追加保留が筆画に沿うか、汚れも保留するかを目視し、`visual_findings.md` に判断を記録する。原画像にない筆画を作る処理ではない。

## 検証と限界

原画像と固定入力・コード・出力のハッシュ、原寸二値マスク、52矩形の原画素とオーバーレイ、朱・保護・汚れ候補の不変、編集と保留の包含・除外を確認する。保存済みの種と近傍からv5を再構成して全21ページの画素一致を確認する。赤のない8ページの不変、既知の濃い楕円状斑点3,131画素の編集対象保持も回帰確認する。ビューアのリンクとJavaScript構文を確認し、ブラウザ実表示は未検証。

元のv4の誤支持や、朱に近い細長い汚れ、確定朱のない孤立筆記、種が弱い太い／一様な筆画は課題。 native-pixelのしきい値は解像度をまたいだ校正ではない。元論文／ローカルCSC／ACEへの総合的優越と未知冊への汎化は未確認。今回の例に合うだけで採用版としない。
''')
    # Reuse the verified viewer behavior and styles; controls and labels belong to v6.
    style=re.search(r'<style>(.*?)</style>',(V5/'index.html').read_text(),re.S).group(1)
    shutil.copyfile(V5/'viewer.js',OUT/'viewer.js')
    controls=''.join(f'<label><input type="checkbox" data-toggle="{key}" {"checked" if key in ["original","v5_edit","split_edit","regained_hold"] else ""}>{html.escape(label)}</label>' for key,label in NAMES.items())
    links=' · '.join(f'<a href="{z["image"]}">{html.escape(z["title"])}</a>' for z in zooms)
    toc=' · '.join(f'<a href="#{r["id"]}">{r["id"]}</a>' for r in cases)
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>暖色保留v6：経路ごとの色条件</title><style>'+style+'</style><header><h1>暖色保留v6：経路ごとの色条件</h1><p>21ページ・20冊・52か所。既知の開発／回帰資料。黄＝編集許可、紫＝v5から増えた保留。線状応答の種を変えずに、筆記側の出力の色条件だけ広げた。保留は文字認定ではなく編集見送り。</p><p><a href="report.md">条件と結果</a> · <a href="verification.json">検証</a> · <a href="../warm_ink_hold_v5_20260930/visual_findings.md">前段の分析</a></p><p>'+links+'</p><nav>'+toc+'</nav></header><div class="controls">'+controls+'<p><label><input type="checkbox" id="native">原寸表示（横スクロール）</label></p></div>'+''.join(cards)+'<script src="viewer.js"></script></html>\n')
    dump(OUT/'verification.json',dict(status='pending'))
    files=[p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts and p.name not in ['complete.json','verification.json','progress.json','run.log']]
    dump(OUT/'complete.json',dict(status='complete',outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(files)}))
    verify();dump(OUT/'progress.json',dict(status='verified',completed_pages=21,total_pages=21))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    verify() if args.verify_only else main()
