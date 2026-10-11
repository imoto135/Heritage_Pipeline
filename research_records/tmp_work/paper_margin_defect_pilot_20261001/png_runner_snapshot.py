"""Apply frozen v6 to new failure pages; deliver PNG comparisons only.

Reuse the validated computation without changing its thresholds. HTML from
intermediate color diagnostics is not used as a comparison deliverable.
"""
import gc
import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import compare_saved_npj_v6 as comparison
from collect_saved_npj_failures import saved_masks
from build_npj_v6_comparison import sha,dump,read,write,overlay,figure

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/npj_failure_followup_20261001'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_cache')
cv2.setNumThreads(1)


def render_png():
    selected=json.loads((OUT/'baseline_review.json').read_text());diagnostics=[]
    for row in selected['cases']:
        im=read(ROOT/row['original']);npj=saved_masks(row)
        background=read(ROOT/row['npj_background'],True)
        assert np.all(np.sum([m.astype(np.uint8) for m in [*npj.values(),background]],axis=0)==1)
        v6={k:read(OUT/'v6_masks'/row['page']/(k+'.png'),True) for k in ['edit','hold','extra','red','protect']}
        assert not (v6['edit']&(v6['red']|v6['protect']|v6['hold'])).any()
        regions=[dict(id=row['page']+'_whole',title='ページ全体',xywh=[0,0,im.shape[1],im.shape[0]])]+row['rois']
        for roi in regions:
            x,y,w,h=roi['xywh'];assert x>=0 and y>=0 and x+w<=im.shape[1] and y+h<=im.shape[0]
            s=(slice(y,y+h),slice(x,x+w));original=im[s]
            baseline=overlay(original,[('edit',npj['edit'][s]),('blue',npj['black'][s]),('red',npj['red'][s])])
            proposal=overlay(original,[('edit',v6['edit'][s]),('hold',v6['hold'][s]),('blue',v6['protect'][s]),('red',v6['red'][s])])
            figure([original,baseline,proposal],['原画像','保存済みnpj v1','固定v6'],OUT/'comparisons'/(roi['id']+'.png'),
                   roi['title']+' · '+row['page'],'青：墨候補/保護　赤：朱候補　黄：編集候補　紫：保留',max_width=620)
            for suffix,value in [('original',original),('npj',baseline),('v6',proposal),
                                 ('npj_edit',npj['edit'][s]),('v6_edit',v6['edit'][s])]:
                path=OUT/'native'/(roi['id']+'_'+suffix+'.png');write(path,value)
                assert np.array_equal(read(path,value.dtype==bool),value)
            diagnostics.append(dict(id=roi['id'],page=row['page'],xywh=roi['xywh'],
                **{'npj_'+k+'_pixels':int(m[s].sum()) for k,m in npj.items()},
                **{'v6_'+k+'_pixels':int(m[s].sum()) for k,m in v6.items()}))
        print('PNG comparisons:',row['page'],flush=True);del im,npj,v6;gc.collect()
    dump(OUT/'comparison_diagnostics.json',diagnostics)
    manifest=json.loads((OUT/'comparison_manifest.json').read_text())
    for relative,digest in manifest['dependencies'].items():assert sha(ROOT/relative)==digest,relative
    outputs={str(p.relative_to(OUT)):sha(p) for folder in ['comparisons','native','v6_masks'] for p in (OUT/folder).rglob('*.png')}
    dump(OUT/'comparison_complete.json',dict(outputs=outputs))
    dump(OUT/'verification.json',dict(status='passed',pages=len(selected['cases']),regions=len(diagnostics),
        frozen_v6=True,saved_npj_not_recomputed=True,input_and_code_hashes_checked=True,
        native_overlay_pixels_equal=True,saved_mask_partition=True,native_binary_masks=True,
        primary_artifact='PNG',pixel_GT=False,restoration=False))


def main():
    comparison.OUT=OUT;comparison.render=render_png
    comparison.compute()
    manifest=json.loads((OUT/'comparison_manifest.json').read_text())
    manifest['dependencies'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    manifest['runner']='scripts/benchmark/run_npj_failure_followup.py'
    manifest['primary_artifact']='PNG'
    dump(OUT/'comparison_manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'png_runner_snapshot.py')
    render_png()


if __name__=='__main__':main()
