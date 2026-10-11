"""Compare saved full-corpus npj v1 masks with frozen v6 on reviewed ROIs."""
from __future__ import annotations
import argparse
import csv
import gc
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/npj_failure_collection_20261001'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'4')
import cv2
import numpy as np
import check_red_positive_fixed as adapter
import prepare_training_free_fixed_check as foreground
import check_warm_hold_other_books as warm
from build_npj_v6_comparison import sha,dump,read,write,figure,overlay
from collect_saved_npj_failures import saved_masks
cv2.setNumThreads(1)


def selection():return json.loads((OUT/'baseline_review.json').read_text())


def compute():
    selected=selection();known={r['id']:r for r in json.loads((warm.v6.OUT/'manifest.json').read_text())['cases']}
    dependencies={}
    for module in [warm.v3,warm.v4,warm.v5,warm.v6]:
        frozen=json.loads((module.OUT/'manifest.json').read_text())
        assert sha(Path(module.__file__))==frozen['source_sha256']
        assert module.SETTINGS==frozen['settings']
        dependencies[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    freeze=json.loads((warm.v6.OUT/'candidate_freeze.json').read_text());assert freeze['variant']=='split_direction'
    for directory,name in [(adapter.COLOR,'compare_training_free_color_shape.py'),
                           (adapter.FROZEN,'refine_training_free_protection.py'),
                           (adapter.PREPROCESS,'compare_training_free_preprocessing.py')]:
        expected=json.loads((directory/'manifest.json').read_text())['source_sha256']
        p=ROOT/'scripts/benchmark'/name;assert sha(p)==expected;dependencies[str(p.relative_to(ROOT))]=expected
    for p in [Path(__file__),Path(adapter.__file__),Path(foreground.__file__),Path(warm.__file__),
              OUT/'baseline_review.json',warm.v6.OUT/'candidate_freeze.json']:
        dependencies[str(p.relative_to(ROOT))]=sha(p)
    for row in selected['cases']:
        for key in ['original','npj_input','npj_black','npj_red','npj_degradation','npj_background']:
            path=ROOT/row[key];dependencies[row[key]]=sha(path)
    dump(OUT/'comparison_manifest.json',dict(dependencies=dependencies,source_sha256=sha(Path(__file__)),
        baseline='Saved full-corpus local npj v1, no ACE/CSC recomputation',
        variant='frozen split_direction v6',native_resolution=True,
        method_thresholds_changed=False,pixel_GT=False,restoration=False,
        selection_before_v6=True,selection_sha256=sha(OUT/'baseline_review.json')))
    shutil.copyfile(__file__,OUT/'comparison_runner_snapshot.py')
    new=[r for r in selected['cases'] if r['page'] not in known]
    pilot=OUT/'v6_new/pilot';review=OUT/'v6_new/input_review';color=OUT/'v6_new/color'
    if new:
        dump(review/'manifest.json',dict(cases=[dict(id=r['page'],title=r['page'],original=r['original']) for r in new],
             inputs={r['original']:sha(ROOT/r['original']) for r in new}))
        dump(review/'rois.json',{r['page']:[dict(title=s['title'],xywh=s['xywh']) for s in r['rois']] for r in new})
        dump(pilot/'manifest.json',dict(settings=json.loads((adapter.PREPROCESS/'manifest.json').read_text())['settings'],
                                       inputs={r['original']:sha(ROOT/r['original']) for r in new}))
        for row in new:
            dest=pilot/row['page']
            if (dest/'complete.json').exists():
                for rel,digest in json.loads((dest/'complete.json').read_text())['outputs'].items():assert sha(pilot/rel)==digest
                continue
            started=time.monotonic();original=read(ROOT/row['original'])
            local,votes,normalized=foreground.foreground_inputs(original)
            for name,value in [('original.png',original),('sauvola_local/foreground.png',local),
                               ('scale_votes.png',votes),('normalized.png',normalized)]:write(dest/name,value)
            dump(dest/'complete.json',dict(outputs={str(p.relative_to(pilot)):sha(p) for p in dest.rglob('*.png')}))
            print('Prepared original:',row['page'],round(time.monotonic()-started,1),'seconds',flush=True)
            del original,local,votes,normalized;gc.collect()
        dump(pilot/'complete.json',dict(outputs={str(p.relative_to(pilot)):sha(p) for p in pilot.rglob('*.png')}))
        if not (color/'complete.json').exists():
            subprocess.run([sys.executable,'-B',str(ROOT/'scripts/benchmark/compare_training_free_color_shape.py'),
                            '--pilot',str(pilot),'--review',str(review),'--output',str(color)],check=True,cwd=ROOT)
        assert json.loads((color/'manifest.json').read_text())['parameters']==json.loads((adapter.COLOR/'manifest.json').read_text())['parameters']
        for rel,digest in json.loads((color/'complete.json').read_text())['outputs'].items():assert sha(color/rel)==digest
    adapter.OUT=OUT;refine=adapter.refinement_function()
    for row in selected['cases']:
        stem=row['page'];dest=OUT/'v6_masks'/stem
        if (dest/'complete.json').exists():
            for name,digest in json.loads((dest/'complete.json').read_text())['outputs'].items():assert sha(dest/name)==digest
            print('Verified cached v6:',stem,flush=True);continue
        started=time.monotonic();original=read(ROOT/row['original'])
        if stem in known:
            old=known[stem];assert np.array_equal(original,read(ROOT/old['original']))
            source=warm.v6.OUT/stem;products=json.loads((warm.v6.OUT/'complete.json').read_text())['outputs']
            maps=dict(edit='split_direction_edit_allowed.png',hold='split_direction_hold.png',extra='split_direction_extra_hold.png',
                      red='red.png',protect='protect.png')
            for name,file in maps.items():
                assert sha(source/file)==products[str((source/file).relative_to(warm.v6.OUT))]
                write(dest/(name+'.png'),read(source/file,True))
            provenance=dict(cached_source=str(source.relative_to(ROOT)),input_pixels_equal=True)
        else:
            ref,components=adapter.refine_inputs(refine,color,pilot,stem)
            values=warm.warm_masks(original,ref,read(color/stem/'red_weak.png',True))
            extra=values[-1]['split_direction'];edit=ref['edit_allowed']&~extra;held=ref['hold']|extra
            assert not (extra&~ref['edit_allowed']).any()
            assert not (edit&held).any()
            for key in ['protect','red','protection_guard']:assert not (edit&ref[key]).any()
            for key,mask in dict(edit=edit,hold=held,extra=extra,red=ref['red'],protect=ref['protect']).items():write(dest/(key+'.png'),mask)
            dump(dest/'features.json',dict(v3=values[1],v4=values[4],v5=values[6]))
            provenance=dict(newly_computed=True,frozen_v5_formula_parity=True)
            del ref,values,components,extra,edit,held;gc.collect()
        dump(dest/'complete.json',dict(**provenance,outputs={p.name:sha(p) for p in dest.glob('*.png')},seconds=time.monotonic()-started))
        print('Fixed v6:',stem,round(time.monotonic()-started,1),'seconds',flush=True)
        del original;gc.collect()
    render()


def render():
    selected=selection();cards=[];diagnostics=[]
    for row in selected['cases']:
        stem=row['page'];im=read(ROOT/row['original']);npj=saved_masks(row)
        v6={k:read(OUT/'v6_masks'/stem/(k+'.png'),True) for k in ['edit','hold','extra','red','protect']}
        for mask in [*npj.values(),*v6.values()]:assert mask.shape==im.shape[:2]
        assert not (v6['edit']&(v6['hold']|v6['protect']|v6['red'])).any()
        assert not (npj['edit']&(npj['black']|npj['red'])).any()
        regions=[dict(id=stem+'_whole',title='ページ全体',xywh=[0,0,im.shape[1],im.shape[0]])]+row['rois']
        parts=[]
        for roi in regions:
            x,y,w,h=roi['xywh'];assert x>=0 and y>=0 and x+w<=im.shape[1] and y+h<=im.shape[0]
            sl=(slice(y,y+h),slice(x,x+w));original=im[sl]
            baseline=overlay(original,[('edit',npj['edit'][sl]),('blue',npj['black'][sl]),('red',npj['red'][sl])])
            proposal=overlay(original,[('edit',v6['edit'][sl]),('hold',v6['hold'][sl]),('blue',v6['protect'][sl]),('red',v6['red'][sl])])
            path=OUT/'comparisons'/(roi['id']+'.png')
            figure([original,baseline,proposal],['原画像','保存済みnpj v1','固定v6'],path,roi['title']+' · '+stem,
                   '青：墨候補/保護　赤：朱候補　黄：編集候補　紫：保留（文字の認定ではありません）',max_width=620)
            for suffix,value in [('original',original),('npj',baseline),('v6',proposal)]:
                write(OUT/'native'/(roi['id']+'_'+suffix+'.png'),value)
            # Native edit masks separately support inspection without color occlusion.
            for suffix,value in [('npj_edit',npj['edit'][sl]),('v6_edit',v6['edit'][sl]),('v6_extra',v6['extra'][sl])]:
                write(OUT/'native'/(roi['id']+'_'+suffix+'.png'),value)
            diagnostics.append(dict(id=roi['id'],page=stem,xywh=roi['xywh'],
                **{'npj_'+k+'_pixels':int(mask[sl].sum()) for k,mask in npj.items()},
                **{'v6_'+k+'_pixels':int(mask[sl].sum()) for k,mask in v6.items()}))
            observation=html.escape(roi.get('baseline_observation',''))
            native=' · '.join(f'<a href="native/{roi["id"]}_{suffix}.png">{title}</a>' for suffix,title in
                            [('original','原画像原寸'),('npj','npj原寸'),('v6','v6原寸'),('npj_edit','npj編集マスク'),('v6_edit','v6編集マスク')])
            parts.append(f'<section id="{roi["id"]}"><h3>{html.escape(roi["title"])}</h3><p>{observation}</p><p>{native}</p><a href="comparisons/{roi["id"]}.png"><img src="comparisons/{roi["id"]}.png" loading="lazy"></a></section>')
        cards.append(f'<article><h2>{stem}</h2><p>{row["split"]} / {row["book"]} · {html.escape(row["evaluation_group"])} · 原寸 {im.shape[1]}×{im.shape[0]}</p>'+''.join(parts)+'</article>')
        print('Rendered:',stem,flush=True);del im,npj,v6;gc.collect()
    dump(OUT/'comparison_diagnostics.json',diagnostics)
    (OUT/'index.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>保存済みnpjの失敗収集と固定v6比較</title><style>body{font-family:system-ui,sans-serif;margin:24px;background:#faf8f3;line-height:1.6}img{width:100%;max-width:1800px}article{margin:60px 0}section{margin:36px 0}</style><h1>保存済みnpjの失敗収集と固定v6比較</h1><p>イラストページを除外。原画像と旧npj出力で選んだ問題候補を、固定v6と同じ位置で比較。画素数はGT精度ではありません。復元・学習・しきい値変更は行っていません。</p><p><a href="report.md">目視結果</a> · <a href="baseline_review.json">v6を見る前の選定</a> · <a href="verification.json">検証</a></p>'+''.join(cards)+'</html>\n')
    verify()


def verify():
    manifest=json.loads((OUT/'comparison_manifest.json').read_text())
    for rel,digest in manifest['dependencies'].items():assert sha(ROOT/rel)==digest,rel
    pages=[]
    for row in selection()['cases']:
        dest=OUT/'v6_masks'/row['page'];shape=read(ROOT/row['original']).shape[:2]
        for name,digest in json.loads((dest/'complete.json').read_text())['outputs'].items():
            p=dest/name;assert sha(p)==digest
            a=cv2.imread(str(p),0);assert a.shape==shape and set(np.unique(a)).issubset({0,255})
        pages.append(row['page'])
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['href','src'] and value and value!='report.md':assert (OUT/value).is_file(),value
    # verification.json itself is written below.
    dump(OUT/'verification.json',dict(status='passed',pages=len(pages),page_ids=pages,
        source_and_input_hashes_checked=True,saved_npJ_not_recomputed=True,
        native_binary_masks=True,mask_polarity_explicit=True,originals_unmodified=True,
        frozen_v6=True,pixel_GT=False,restoration=False,browser_rendering_tested=False))
    Links().feed((OUT/'index.html').read_text())
    dump(OUT/'comparison_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for folder in ['comparisons','native','v6_masks'] for p in (OUT/folder).rglob('*.png')}))
    print('Verification passed:',len(pages),'pages',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['compute','render','verify']);args=p.parse_args()
    {'compute':compute,'render':render,'verify':verify}[args.stage]()


if __name__=='__main__':main()
