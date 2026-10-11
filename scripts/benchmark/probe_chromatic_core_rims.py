"""Two source scopes for original chromatic rims; fixed6px, frozen guards."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import diagnose_chromatic_repair_outlines as diagnostic
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=diagnostic.ROOT
OUT=ROOT/'tmp_work/chromatic_core_rim_mask_pilot_20261001'
MODES=['all6','strong6']
SETTINGS=dict(base='frozen pale6 + short_upper',radius_px=6,
    original_color='same diagnostic delta_a1/delta_b2/absolute_b5/Ldrop>=-2',
    all6='all frozen accepted cores, including thin3.5 and anchored small fragments',
    strong6='original peak6 cores plus original top completions plus short upper strong cores',
    only_new_rim_source_changes=True,old_edits_retained=True,
    domain_guards_fixed=True,one_pass=True,ROI_gating=False,development=True,
    pages=25,pixel_GT=False,restoration=False,new_DDRM=False)


def strong_core(desc):
    thin=diagnostic.prior.mask.pale.prior.anchor.prior.mask;key=desc['page']['page'];source=desc['source']
    if source in thin.prior.MARGIN.SOURCES+[thin.new.INPUT]:
        path=thin.new.OUT/'pages'/key/'features/rim_core.png' if source==thin.new.INPUT else thin.prior.OUT/'pages'/key/'core.png'
        paths=[path];core=read(path,True)
    else:
        new3=diagnostic.prior.mask.pale.prior.new3;new4=diagnostic.prior.mask.pale.prior.new4
        module=new3 if source==new3.INPUT else new4;assert source==module.INPUT
        paths=[module.OUT/'pages'/key/'features'/name for name in ['strict_core.png','completed_core.png']]
        core=read(paths[0],True)|read(paths[1],True)
    path=diagnostic.prior.mask.OUT/'pages'/key/'short_upper/strong_core.png';paths.append(path)
    return core|read(path,True),paths


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(diagnostic.OUT)
    inputs=diagnostic.prior.mask.pale.prior.descriptors();rows=json.loads((diagnostic.prior.OUT/'diagnostics.json').read_text())
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [diagnostic.OUT,diagnostic.prior.OUT]:
        for name in ['manifest.json','complete.json']:deps[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for r in inputs:
        deps[r['page']['original']]=sha(ROOT/r['page']['original'])
        _,paths=strong_core(r)
        for path in paths:deps[str(path.relative_to(ROOT))]=sha(path)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');pages=[];regions=[];external=[]
    for index,desc in enumerate(inputs,1):
        key=desc['page']['page'];original=read(ROOT/desc['page']['original']);source=diagnostic.OUT/'pages'/key;dest=OUT/'pages'/key
        base=read(source/'edit.png',True);core=read(source/'core.png',True);domain=read(source/'domain.png',True);guard=read(source/'guard.png',True)
        warm=read(source/'warm_support.png',True);strong,_=strong_core(desc);assert not np.any(strong&~core)
        additions={};covered=np.zeros(base.shape,bool)
        for row in [r for r in rows if r['page']==key]:
            x,y,w,h=row['xywh'];covered[y:y+h,x:x+w]=True
        for mode,seed in [('all6',core),('strong6',strong)]:
            near=cv2.dilate(seed.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
            add=near&~base&domain&~guard&warm;additions[mode]=add
            if mode=='all6':assert np.array_equal(add,read(source/'warm6.png',True))
            assert not np.any(add&(base|~domain|guard))
            for name,item in dict(edit=base|add,added=add,source_core=seed).items():write(dest/mode/(name+'.png'),item)
        assert not np.any(additions['strong6']&~additions['all6'])
        outside=additions['all6']&~covered
        # Include every core component whose chromatic rim falls outside existing73 ROI rectangles.
        n,labels,stats,_=cv2.connectedComponentsWithStats(core.astype(np.uint8),connectivity=8)
        extra_covered=np.zeros_like(base)
        for k in range(1,n):
            x,y,w,h,area=map(int,stats[k]);x0,y0=max(0,x-6),max(0,y-6);x1,y1=min(base.shape[1],x+w+6),min(base.shape[0],y+h+6)
            if not outside[y0:y1,x0:x1].any():continue
            local=labels[y0:y1,x0:x1]==k
            halo=cv2.dilate(local.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))>0
            touching=halo&outside[y0:y1,x0:x1]
            if not touching.any():continue
            rx,ry=max(0,x-45),max(0,y-45);rw,rh=min(base.shape[1],x+w+45)-rx,min(base.shape[0],y+h+45)-ry
            extra_covered[ry:ry+rh,rx:rx+rw]=True
            external.append(dict(id=key+'_external_rim_core_'+str(k),page=key,original=desc['page']['original'],
                title='固定ROI外の暖色輪郭、原因未確定 '+str(k),xywh=[rx,ry,rw,rh],
                source_component=dict(id=k,xywh=[x,y,w,h],area=area),outside_previous_ROIs=int(touching.sum()),
                selection_after_mask_outputs=True,pixel_GT=False))
        assert not np.any(outside&~extra_covered)
        for name,item in dict(base_edit=base,domain=domain,guard=guard,core=core,strong_core=strong,existing_ROI_coverage=covered,outside_existing_ROIs=outside).items():write(dest/(name+'.png'),item)
        pages.append(dict(page=key,additions={m:int(v.sum()) for m,v in additions.items()},outside_existing_ROIs=int(outside.sum())))
        roi_rows=[dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+[r for r in rows if r['page']==key]+[r for r in external if r['page']==key]
        for row in roi_rows:
            rid=row['id'];x,y,w,h=row['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s];before=crop.copy();before[base[s]]=(0,200,255)
            panels=[]
            for mode in MODES:
                show=before.copy();show[additions[mode][s]]=(255,120,0);panels.append(show)
            figure([crop,before]+panels,['原画像','黄＝固定編集','青＝全コア6pxの色差輪郭','青＝強コア6pxの色差輪郭'],OUT/'comparisons'/(rid+'.png'),
                row['title']+' · 狭い暖色外縁、元コアの強さを比較','既存編集・安全領域・保護を固定。青は新しい追加候補、復元は別途比較。',
                '半径6pxは固定。一段の観測色差だけ。12px拡張・低次紙色面は併用しない。')
            regions.append(dict(id=rid,page=key,xywh=row['xywh'],additions={m:int(v[s].sum()) for m,v in additions.items()}))
        print('Chromatic core rims:',index,'/25',key,pages[-1],flush=True)
        del original,additions,labels;gc.collect()
    for path,value in deps.items():assert sha(ROOT/path)==value
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=regions,external_regions=external))
    dump(OUT/'verification.json',dict(status='passed',pages=25,all6_matches_diagnostic=True,strong6_subset=True,
        outside_existing_ROIs_all_covered_in_external_regions=True,old_edits_preserved=True,guards_domain_preserved=True,
        restoration=False,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:chromatic rim masks; external regions',len(external),flush=True)


if __name__=='__main__':main()
