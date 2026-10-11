"""Isolate the v5 ridge gate using exactly the same color and context masks."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess

from compare_warm_stroke_hold import ROOT, V4, OUT as V5, read, write, dump, sha, overlay, COLORS
import cv2
import numpy as np

OUT=ROOT/'tmp_work/warm_stroke_ablation_v5_20260930'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
SETTINGS=dict(no_ridge='v5 color_direction & v2 edit_allowed & (v5 near_red | v4 extra_hold)',
              with_ridge='Frozen v5 stroke_context_extra_hold',
              shared='Same pixels, direction18 degrees, red distance96, v4 support; remove only the ridge requirement',
              purpose='Development ablation: ridge-reopened edits may be either stains or real ink; no pixel GT')


def verify():
    m=json.loads((OUT/'manifest.json').read_text()); assert m['source_sha256']==sha(Path(__file__)) and m['settings']==SETTINGS
    for rel,h in m['dependencies'].items():assert sha(ROOT/rel)==h,rel
    c=json.loads((OUT/'complete.json').read_text())
    for rel,h in c['outputs'].items():assert sha(OUT/rel)==h,rel
    negative=[]
    for row in m['cases']:
        stem=row['id'];dest=OUT/stem;original=read(V5/stem/'original.png')
        color=read(V5/stem/'color_direction.png',True);near=read(V5/stem/'near_red.png',True)
        prior=read(V4/stem/'red_protect_context_extra_hold.png',True)
        old=read(ROOT/row['reference']/'edit_allowed.png',True)
        actual=read(V5/stem/'stroke_context_edit_allowed.png',True)
        expected=color&old&(near|prior);extra=read(dest/'no_ridge_extra_hold.png',True)
        edit=read(dest/'no_ridge_edit_allowed.png',True);reopened=read(dest/'ridge_reopened_edit.png',True)
        assert np.array_equal(extra,expected) and np.array_equal(edit,old&~extra)
        assert not (edit&~actual).any()
        assert np.array_equal(reopened,actual&~edit)
        if row['group']=='negative_regression':assert not extra.any();negative.append(stem)
        for p in dest.glob('*.png'):
            v=cv2.imread(str(p),0);assert v.shape==original.shape[:2] and set(np.unique(v)).issubset({0,255})
        for i,roi in enumerate(row['rois'],1):
            x,y,w,h=roi['xywh'];sl=(slice(y,y+h),slice(x,x+w))
            for key,mask in [('no_ridge_edit',edit),('ridge_reopened_edit',reopened)]:
                assert np.array_equal(read(dest/'review'/f'roi_{i:02d}_{key}.png'),overlay(original[sl],mask[sl],COLORS['edit']))
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['src','href'] and v and not v.startswith('#'):assert (OUT/v).is_file(),v
    Links().feed((OUT/'index.html').read_text())
    subprocess.run(['node','--check',str(OUT/'viewer.js')],check=True,cwd=ROOT)
    dump(OUT/'verification.json',dict(status='passed',pages=21,books=20,rois=52,
         output_hashes_checked=len(c['outputs']),matched_context_color_checked=True,
         with_ridge_extra_subset_of_no_ridge=True,native_binary_and_overlay_pixels_checked=True,
         negative_pages_unchanged=negative,viewer_links_and_javascript_checked=True,
         browser_rendering_tested=False,pixel_GT=False,independent_accuracy_test=False))
    print('Ablation verification passed',flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    parent=json.loads((V5/'manifest.json').read_text());cases=parent['cases']
    hashes={};records={V5:json.loads((V5/'complete.json').read_text())['outputs'],V4:json.loads((V4/'complete.json').read_text())['outputs']}
    for p in [V5/'manifest.json',Path(__file__).with_name('compare_warm_stroke_hold.py')]:hashes[str(p.relative_to(ROOT))]=sha(p)
    for row in cases:
        for base,names in [(V5,['original.png','color_direction.png','near_red.png','stroke_context_edit_allowed.png','stroke_context_extra_hold.png']),
                           (V4,['red_protect_context_extra_hold.png'])]:
            for name in names:
                p=base/row['id']/name;expected=records[base][str(p.relative_to(base))];assert sha(p)==expected
                hashes[str(p.relative_to(ROOT))]=expected
        p=ROOT/row['reference']/'edit_allowed.png';h=parent['dependencies'][str(p.relative_to(ROOT))];assert sha(p)==h;hashes[str(p.relative_to(ROOT))]=h
    hashes[str((V5/'failure_zoom_diagnostics.json').relative_to(ROOT))]=sha(V5/'failure_zoom_diagnostics.json')
    manifest=dict(source_sha256=sha(Path(__file__)),settings=SETTINGS,cases=cases,dependencies=hashes)
    mp=OUT/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('Use a new version for changed inputs/code')
    dump(mp,manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py')
    metrics=[];cards=[]
    for row in cases:
        stem=row['id'];dest=OUT/stem;original=read(V5/stem/'original.png');h,w=original.shape[:2]
        color=read(V5/stem/'color_direction.png',True);near=read(V5/stem/'near_red.png',True)
        prior=read(V4/stem/'red_protect_context_extra_hold.png',True);old=read(ROOT/row['reference']/'edit_allowed.png',True)
        actual=read(V5/stem/'stroke_context_edit_allowed.png',True)
        extra=color&old&(near|prior);edit=old&~extra;reopened=actual&~edit
        for name,mask in [('no_ridge_extra_hold',extra),('no_ridge_edit_allowed',edit),('ridge_reopened_edit',reopened)]:write(dest/(name+'.png'),mask)
        parts=[]
        rects=[(None,'ページ全体',[0,0,w,h])]+[(i,r['title'],r['xywh']) for i,r in enumerate(row['rois'],1)]
        for i,title,box in rects:
            x,y,rw,rh=box;sl=(slice(y,y+rh),slice(x,x+rw))
            metrics.append(dict(page=stem,group=row['group'],book=row['book'],roi=i,title=title,xywh=box,
                                no_ridge_edit_pixels=int(edit[sl].sum()),v5_edit_pixels=int(actual[sl].sum()),ridge_reopened_pixels=int(reopened[sl].sum())))
            if i is None:continue
            panels=[]
            for key,label in [('original','原画像'),('no_ridge_edit','線状応答条件なし：編集許可'),('stroke_edit','線状応答条件あり：編集許可'),('ridge_reopened_edit','線状応答条件によって編集へ戻る部分')]:
                if key in ['original','stroke_edit']:rel=f'../{V5.name}/{stem}/review/roi_{i:02d}_{key}.png'
                else:
                    rel=f'{stem}/review/roi_{i:02d}_{key}.png';write(OUT/rel,overlay(original[sl],(edit if key=='no_ridge_edit' else reopened)[sl],COLORS['edit']))
                panels.append(f'<figure data-mode="{key}"><figcaption>{label}</figcaption><a href="{rel}"><img src="{rel}" loading="lazy" alt="{label}"></a></figure>')
            parts.append(f'<section><h3>{html.escape(title)}</h3><div class="grid">{"".join(panels)}</div></section>')
        cards.append(f'<article><h2>{stem}</h2>{"".join(parts)}</article>')
        print(stem+' ablation computed',flush=True)
    dump(OUT/'diagnostics.json',metrics)
    zooms=[]
    for i,z in enumerate(json.loads((V5/'failure_zoom_diagnostics.json').read_text()),1):
        stem=z['page'];x,y,w,h=z['xywh'];sl=(slice(y,y+h),slice(x,x+w));original=read(V5/stem/'original.png')[sl]
        no=read(OUT/stem/'no_ridge_edit_allowed.png',True)[sl];yes=read(V5/stem/'stroke_context_edit_allowed.png',True)[sl]
        returned=read(OUT/stem/'ridge_reopened_edit.png',True)[sl]
        fig,axes=plt.subplots(1,4,figsize=(15.,max(2.,min(8.,3.75*h/w+.7))))
        for ax,key,mask,label in zip(axes,['original','no_ridge','v5','returned'],[None,no,yes,returned],['原画像','線状応答条件なし','線状応答条件あり','編集へ戻る部分']):
            picture=original if mask is None else overlay(original,mask,COLORS['edit']);write(OUT/'failure_zooms'/f'zoom_{i:02d}_{key}.png',picture)
            ax.imshow(cv2.cvtColor(picture,cv2.COLOR_BGR2RGB),interpolation='nearest');ax.set_title(label,fontproperties=font,fontsize=10);ax.axis('off')
        fig.suptitle(z['title']+'（同じ色・文脈条件、GTなし）',fontproperties=font);fig.tight_layout(rect=(0,0,1,.92))
        rel=f'failure_zooms/zoom_{i:02d}_comparison.png';fig.savefig(OUT/rel,dpi=145,facecolor='white');plt.close(fig)
        zooms.append(dict(page=stem,title=z['title'],xywh=z['xywh'],image=rel,no_ridge_edit_pixels=int(no.sum()),v5_edit_pixels=int(yes.sum()),ridge_reopened_pixels=int(returned.sum()),pixel_GT=False))
    dump(OUT/'failure_zoom_diagnostics.json',zooms)
    full=[m for m in metrics if m['roi'] is None]
    dump(OUT/'summary.json',{k:sum(r[k] for r in full) for k in ['no_ridge_edit_pixels','v5_edit_pixels','ridge_reopened_pixels']})
    table='\n'.join(f'| [{z["title"]}]({z["image"]}) | {z["no_ridge_edit_pixels"]:,} | {z["v5_edit_pixels"]:,} | {z["ridge_reopened_pixels"]:,} |' for z in zooms)
    (OUT/'report.md').write_text(f'''# v5の局所線状応答だけを外す比較

色方向±18度、朱から96px以内、v4の保留支持、入力画素を全てそろえ、線状応答を必要とする条件だけを外した。線状応答条件なしの追加保留は `v5色条件 & v2編集候補 & (朱96px近傍 | v4追加保留)`。v5の保留はこの部分集合であり、線状条件によって保留を減らすと編集へ戻る画素が生じる。

全21ページ・20冊・52矩形を計算した。原画像・保護・朱・汚れ候補は変更せず、学習・画素GT・独立精度評価・画像削除は行っていない。

| 同じ診断矩形 | 線状応答条件なしの編集候補 | v5編集候補 | 線状応答条件で編集へ戻る画素 |
|---|---:|---:|---:|
{table}

文字に戻った編集候補は悪化、汚れに戻った編集候補は望ましい可能性がある。画素GTがないため、戻った数を誤検出削減量やPrecisionとして扱わない。v5内の「短距離の朱文脈だけ」と「線状変化も利用」はv4支持の有無も違うため、その2列だけから線状特徴の効果を断定しない。この比較で条件をそろえて確認する。

元の出力のハッシュ、同じ色・文脈条件、二値原寸マスク、52矩形の表示画素、v5追加保留の部分集合関係を検証する。赤のない8ページは両方式で追加保留なし。ビューアのリンクとJavaScript構文を確認し、ブラウザ実表示は未検証。目視の判断は親実験v5の所見へ記録する。
''')
    links=' · '.join(f'<a href="{z["image"]}">{html.escape(z["title"])}</a>' for z in zooms)
    (OUT/'viewer.js').write_text("const cs=[...document.querySelectorAll('[data-toggle]')];function u(){for(const c of cs)for(const f of document.querySelectorAll('figure[data-mode=\"'+c.dataset.toggle+'\"]'))f.hidden=!c.checked;}cs.forEach(c=>c.addEventListener('change',u));document.querySelector('#native').addEventListener('change',e=>document.body.classList.toggle('native',e.target.checked));u();\n")
    controls=''.join(f'<label><input type="checkbox" data-toggle="{k}" checked>{l}</label>' for k,l in [('original','原画像'),('no_ridge_edit','線状条件なし'),('stroke_edit','線状条件あり'),('ridge_reopened_edit','編集へ戻る部分')])
    (OUT/'index.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v5線状条件の比較</title><style>body{font-family:system-ui,sans-serif;margin:24px;background:#f3f1ed;line-height:1.7}article,header{background:white;padding:20px;margin:20px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:15px}figure{margin:0}img{width:100%;height:auto}figure[hidden]{display:none}label{margin-right:20px}body.native .grid{display:flex;overflow:auto}body.native figure{flex:none}body.native img{width:auto}</style><header><h1>同じ色・文脈で線状応答の条件だけを外す比較</h1><p>21ページ・20冊・52か所。GTなしの開発資料。黄＝編集候補。線状条件で汚れが編集へ戻るか、筆画まで戻るかを確認する。</p><p><a href="report.md">数値と条件</a> · <a href="verification.json">検証</a> · <a href="../warm_ink_hold_v5_20260930/index.html">親実験v5</a></p><p>'''+links+'</p>'+controls+'<label><input type="checkbox" id="native">原寸表示</label></header>'+''.join(cards)+'<script src="viewer.js"></script></html>\n')
    dump(OUT/'verification.json',dict(status='pending'))
    files=[p for p in OUT.rglob('*') if p.is_file() and 'plot_cache' not in p.parts and p.name not in ['complete.json','verification.json','run.log']]
    dump(OUT/'complete.json',dict(status='complete',outputs={str(p.relative_to(OUT)):sha(p) for p in sorted(files)}));verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    verify() if args.verify_only else main()
