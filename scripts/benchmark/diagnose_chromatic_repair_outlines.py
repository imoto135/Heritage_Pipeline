"""Separate unedited chromatic rims from already filled paper, no mask mutation."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import restore_full_upper_strong as prior
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/chromatic_repair_outline_diagnostic_20261001'
RADII=[6,12]
SETTINGS=dict(baseline='frozen pale6 + short_upper',pages=25,ROIs=73,
    original_features='median3 Lab minus same bright-paper estimate',
    warm_support='delta_a>=1, delta_b>=2, absolute_b>=5, Ldrop>=-2',
    proximity='6 or12px from frozen accepted cores including short upper strong cores',
    candidate='outside frozen edit, same domain, same guards',
    interpretation='diagnostic only; visible color is not a damage GT',
    internal='first3px of retained pure-paper applied map, color difference to paper proxy',
    masks_changed=False,restoration_changed=False,new_DDRM=False,pixel_GT=False,development=True)


def feature_quantiles(values,mask):
    if not mask.any():return None
    return np.percentile(values[mask],[10,50,90],axis=0).tolist()


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT);check_bundle(prior.mask.OUT)
    inputs=prior.mask.pale.prior.descriptors();rows=json.loads((prior.OUT/'diagnostics.json').read_text())
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    estimator=prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.paper_estimate
    estimator_path=Path(prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.__file__)
    dependencies[str(estimator_path.relative_to(ROOT))]=sha(estimator_path)
    for folder in [prior.OUT,prior.mask.OUT]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for r in inputs:dependencies[r['page']['original']]=sha(ROOT/r['page']['original'])
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');pages=[];regions=[]
    for index,r in enumerate(inputs,1):
        pg=r['page'];key=pg['page'];original=read(ROOT/pg['original'])
        median=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        paper=estimator(median);difference=median-paper;drop=-difference[:,:,0]
        folder=prior.mask.OUT/'pages'/key;dest=OUT/'pages'/key
        edit=read(folder/'short_upper/edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True)
        core=read(folder/'core.png',True)|read(folder/'short_upper/strong_core.png',True)
        warm=(difference[:,:,1]>=1)&(difference[:,:,2]>=2)&(median[:,:,2]>=5)&(drop>=-2)
        values={};notes={}
        for radius in RADII:
            near=cv2.dilate(core.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
            candidates=near&~edit&domain&~guard&warm;values[radius]=candidates
            guarded=near&~edit&domain&guard&warm
            below_light2=candidates&(drop<2)
            notes[str(radius)]=dict(unedited_warm=int(candidates.sum()),guarded_warm_retained=int(guarded.sum()),
                below_previous_rim_drop2=int(below_light2.sum()),
                original_Lab_delta_quantiles=feature_quantiles(difference,candidates))
            write(dest/f'warm{radius}.png',candidates);write(dest/f'guarded_warm{radius}.png',guarded)
        assert not np.any(values[6]&~values[12])
        for name,item in dict(edit=edit,core=core,domain=domain,guard=guard,warm_support=warm).items():write(dest/(name+'.png'),item)
        dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/'original_features.npz',median_Lab=median,paper_estimate_Lab=paper)
        pages.append(dict(page=key,diagnostics=notes))
        whole=original.copy();whole[edit]=(0,200,255);shows=[]
        for radius in RADII:
            show=whole.copy();show[values[radius]]=(255,120,0);shows.append(show)
        figure([original,whole]+shows,['原画像','黄＝固定編集','青＝コア6px内の暖色残存','青＝コア12px内の暖色残存'],OUT/'comparisons'/(key+'_whole.png'),
            'ページ全体 · 補修輪郭の取り残し診断','青は元画像の暖色差が残る未編集画素。新しいマスクへ採用していない。',
            '安全領域・保護を維持。暖色だから損傷と断定しない。復元・DDRM変更なし。')
        for row in [z for z in rows if z['page']==key]:
            rid=row['id'];x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];s=(slice(y,y+h),slice(x,x+w))
            context_slice=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w));native=prior.OUT/'native'/rid/'short_upper'
            crop=original[s];base=read(native/'output.png');applied=read(native/'retained_applied_context.png',True)
            distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
            inside=(applied&(distance<=3))[context_slice]
            basemedian=cv2.cvtColor(cv2.medianBlur(base,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            outside={str(k):dict(count=int(v[s].sum()),original_Lab_delta_quantiles=feature_quantiles(difference[s],v[s])) for k,v in values.items()}
            panels=[]
            for radius in RADII:
                show=crop.copy();show[inside]=(40,200,40);show[values[radius][s]]=(255,120,0);panels.append(show)
            elsewhere='旧固定14ページの紙面外は保存DDRM。' if row['base_elsewhere']=='saved_DDRM' else '紙面外は原画像。'
            figure([crop,base]+panels,['原画像','固定マスクの補修結果','青＝6px未編集、緑＝補間内縁','青＝12px未編集、緑＝補間内縁'],OUT/'comparisons'/(rid+'.png'),
                row['title']+' · 残存原画素と補間内を分ける',
                '青は原画像の色差、緑は補間済みの内縁3px。候補の採用・補間変更はない。',
                elsewhere+' 紙色推定は観測代理値でGTではない。')
            regions.append(dict(id=rid,page=key,xywh=row['xywh'],outside=outside,filled_inside_edge_pixels=int(inside.sum()),
                original_inside_Lab_delta_quantiles=feature_quantiles(difference[s],inside),
                output_inside_Lab_delta_quantiles=feature_quantiles(basemedian-paper[s],inside)))
        print('Chromatic repair outlines:',index,'/25',key,{k:v['unedited_warm'] for k,v in notes.items()},flush=True)
        del original,median,paper,difference,values;gc.collect()
    assert len(regions)==73
    for path,value in dependencies.items():assert sha(ROOT/path)==value
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=regions))
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=73,mask_and_restoration_changes=0,
        full_page_inference_before_ROIs=True,guards_domain_unchanged=True,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:98 comparison PNGs, chromatic-outline diagnostic only',flush=True)


if __name__=='__main__':main()
