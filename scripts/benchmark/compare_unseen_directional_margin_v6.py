"""Compute frozen v6 and render PNG-only classification comparisons."""
import gc
import json
from pathlib import Path
import shutil
import compare_saved_npj_v6 as driver
from collect_saved_npj_failures import saved_masks
from build_npj_v6_comparison import dump,read,sha,overlay
from check_paper_margin_defect_ddrm import figure

ROOT=driver.ROOT
OUT=ROOT/'tmp_work/unseen_directional_margin_validation_20261001'


def render():
    selected=json.loads((OUT/'baseline_review.json').read_text());rows=[]
    manifest=json.loads((OUT/'comparison_manifest.json').read_text())
    manifest['dependencies'][str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    manifest.update(primary_artifact='PNG',original_selection_first=True)
    dump(OUT/'comparison_manifest.json',manifest);shutil.copyfile(__file__,OUT/'png_runner_snapshot.py')
    for pg in selected['cases']:
        key=pg['page'];original=read(ROOT/pg['original']);npj=saved_masks(pg)
        v6={k:read(OUT/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        for roi in [dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+pg['rois']:
            x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s]
            before=overlay(crop,[('edit',npj['edit'][s]),('blue',npj['black'][s]),('red',npj['red'][s])])
            after=overlay(crop,[('edit',v6['edit'][s]),('hold',v6['hold'][s]),('blue',v6['protect'][s]),('red',v6['red'][s])])
            figure([crop,before,after],['原画像','保存済みnpj','固定v6'],OUT/'classification_comparisons'/(roi['id']+'.png'),roi['title']+' · '+key,
                '青＝墨・保護、赤＝朱、黄＝編集、紫＝保留。マスクだけの比較。','復元・DDRM後の画像ではない。条件変更なし。')
            rows.append(dict(id=roi['id'],page=key,xywh=roi['xywh'],npj_edit=int(npj['edit'][s].sum()),v6_edit=int(v6['edit'][s].sum())))
        del original,npj,v6;gc.collect()
    for n,h in manifest['dependencies'].items():assert sha(ROOT/n)==h
    dump(OUT/'classification_diagnostics.json',rows)
    dump(OUT/'comparison_complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for folder in ['v6_masks','classification_comparisons'] for p in (OUT/folder).rglob('*.png')}))
    print('Complete: fixed v6 classification17 PNGs,4 original-first books',flush=True)


if __name__=='__main__':
    driver.OUT=OUT;driver.render=render;driver.compute()
