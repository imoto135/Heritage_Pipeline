"""Measure original lighter connected support around frozen accepted cores."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_anchored_thin_fragments as anchor
import validate_unseen_thin_margin_candidate_v2 as new3
import validate_unseen_dark_margin_candidate as new4
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=anchor.ROOT
OUT=ROOT/'tmp_work/anchored_pale_support_diagnostic_20261001'
LEVELS=[10,6,4]


def descriptors():
    rows=[];thin=anchor.prior.mask
    for source in thin.prior.MARGIN.SOURCES+[thin.new.INPUT]:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key=page['page'];folder=thin.OUT/'pages'/key
            if source==thin.new.INPUT: core=read(thin.new.OUT/'pages'/key/'features/rim_core.png',True)
            else:core=read(thin.prior.OUT/'pages'/key/'core.png',True)
            core|=read(folder/'new_core.png',True)|read(anchor.OUT/'pages'/key/'near8/fragment_core.png',True)
            rows.append(dict(page=page,source=source,core=core,edit=read(anchor.OUT/'pages'/key/'near8/edit.png',True),
                             domain=read(folder/'domain.png',True),guard=read(folder/'guard.png',True)))
    for mod in [new3,new4]:
        for page in json.loads((mod.INPUT/'baseline_review.json').read_text())['cases']:
            key=page['page'];folder=mod.OUT/'pages'/key/'features'
            components=json.loads((mod.OUT/'pages'/key/'components.json').read_text());assert not components['near8']
            core=read(folder/'strict_core.png',True)|read(folder/'completed_core.png',True)|read(folder/'new_core.png',True)
            rows.append(dict(page=page,source=mod.INPUT,core=core,edit=read(mod.OUT/'pages'/key/'near8/edit.png',True),
                             domain=read(folder/'safe_domain.png',True),guard=read(folder/'rim_guard.png',True)))
    assert len(rows)==25
    return rows


def measure(original,features,core,edit,domain,guard,threshold):
    H,W=domain.shape
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    support=(features['light_drop']>=threshold)&(lab[:,:,0]<=85)&domain&~guard
    connected=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    connected&=domain&~guard
    n,labels,stats,_=cv2.connectedComponentsWithStats(connected.astype(np.uint8),connectivity=8)
    overlap=np.bincount(labels.ravel(),weights=core.ravel(),minlength=n)
    warm=np.bincount(labels.ravel(),weights=features['color_seed'].ravel(),minlength=n)
    inner=cv2.erode(domain.astype(np.uint8),np.ones((3,3),np.uint8))>0
    boundary=np.bincount(labels.ravel(),weights=(connected&~inner).ravel(),minlength=n)
    selected=np.zeros(n,bool);notes=[]
    for i in range(1,n):
        if overlap[i]<32:continue
        x,y,w,h,area=map(int,stats[i]);shape=128<=area<=.01*H*W and max(w,h)<=.2*min(H,W)
        selected[i]=shape
        component=labels==i
        notes.append(dict(id=i,xywh=[x,y,w,h],area=area,core_overlap=int(overlap[i]),warm_seeds=int(warm[i]),
                          boundary_contacts=int(boundary[i]),shape_ok=bool(shape),
                          unedited=int((component&~edit).sum()),
                          scope='diagnostic; boundary-touching candidates are shown, not accepted as a new method'))
    candidate=selected[labels]&connected
    return candidate,notes


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in [anchor.OUT,new3.OUT,new4.OUT]:check_bundle(folder)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [anchor.OUT,new3.OUT,new4.OUT]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for module in [anchor.prior.mask.prior.MARGIN,anchor.prior.mask.prior.MARGIN.second,
                   anchor.prior.mask.prior.MARGIN.first]:dependencies[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    rows=descriptors();pages=[];regions=[]
    for r in rows:
        pg=r['page'];dependencies[pg['original']]=sha(ROOT/pg['original'])
        dependencies[str((r['source']/'baseline_review.json').relative_to(ROOT))]=sha(r['source']/'baseline_review.json')
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=dependencies,pages=25,
                  thresholds=LEVELS,mask_changes=False,restoration=False,pixel_GT=False,
                  purpose='identify disconnected dark support; no method adoption or ROI gating')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    for i,r in enumerate(rows,1):
        page=r['page'];key=page['page'];core=r['core'];edit=r['edit'];domain=r['domain'];guard=r['guard']
        original=read(ROOT/page['original']);source=r['source'];values={};details={}
        if core.any():
            v6={k:read(source/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
            features,_=anchor.prior.mask.prior.MARGIN.load_features(key,original,v6)
            for level in LEVELS:
                values[level],details[level]=measure(original,features,core,edit,domain,guard,level)
            del features,v6
        else:
            for level in LEVELS:values[level]=np.zeros_like(core);details[level]=[]
        dest=OUT/'pages'/key
        for name,item in dict(core=core,edit=edit,domain=domain,guard=guard).items():write(dest/(name+'.png'),item)
        for level in LEVELS:write(dest/f'drop{level}.png',values[level])
        pages.append(dict(page=key,cores=int(core.sum()),diagnostics={str(k):dict(unedited=int((v&~edit).sum()),components=details[k]) for k,v in values.items()}))
        rois=[dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+page['rois']
        for roi in rois:
            rid=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s];shows=[]
            for level in LEVELS:
                show=crop.copy();show[edit[s]]=(0,200,255);show[(values[level]&~edit)[s]]=(255,120,0);shows.append(show)
            figure([crop]+shows,['原画像','暗さ差10の連結診断','暗さ差6の連結診断','暗さ差4の連結診断'],
                   OUT/'comparisons'/(rid+'.png'),roi['title']+' · 受理コアに連なる未編集部分',
                   '黄＝固定編集、青＝未編集の連結支持。青は復元対象に採用していない。',
                   '同じ安全領域・保護。境界接触も可視化する診断のみ。色や損傷の正解判定ではない。')
            regions.append(dict(id=rid,page=key,xywh=roi['xywh'],unedited={str(k):int((v&~edit)[s].sum()) for k,v in values.items()}))
        print('Pale-support diagnostic:',i,'/25',key,pages[-1]['diagnostics']['4']['unedited'],flush=True)
        del original,values,details;gc.collect()
    for p,d in dependencies.items():assert sha(ROOT/p)==d
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=regions))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.py'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete:25-page diagnostic only, no changed masks or restoration',flush=True)


if __name__=='__main__':main()
