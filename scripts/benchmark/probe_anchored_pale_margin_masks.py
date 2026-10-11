"""Grow lighter support connected to frozen cores; compare Ldrop6 and4 only."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import diagnose_anchored_pale_paper_support as prior
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/anchored_pale_margin_mask_pilot_20261001'
LEVELS=[6,4]
SETTINGS=dict(base='frozen thin3.5 + near8 where applicable; original margin/top/rim retained',
    support_Ldrop=LEVELS,support_Lmax=80,support_close=3,minimum_frozen_core_overlap=32,
    component_area=[128,'0.01*page_area'],max_side='0.20*short_page_side',
    top_domain_contact_allowed=True,other_domain_boundary_contact=False,
    core_frozen=True,recursive_chaining=False,halo_added=False,
    domain_and_guards_frozen=True,ROI_gating=False,pages=25,development=True,
    pixel_GT=False,DDRM=False,restoration=False,
    old_unknown_stroke_like_core_candidates_retained=True)


def candidate(original,features,core,domain,guard,level):
    H,W=domain.shape
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    support=(features['light_drop']>=level)&(lab[:,:,0]<=80)&domain&~guard
    connected=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    connected&=domain&~guard
    n,labels,stats,_=cv2.connectedComponentsWithStats(connected.astype(np.uint8),connectivity=8)
    overlap=np.bincount(labels.ravel(),weights=core.ravel(),minlength=n)
    inner=cv2.erode(domain.astype(np.uint8),np.ones((3,3),np.uint8))>0
    allowed_top=np.zeros_like(domain)
    if domain.any():
        yy,xx=np.nonzero(domain);allowed_top[int(yy.min()),int(xx.min())+1:int(xx.max())]=True
    border=connected&~inner
    other=np.bincount(labels.ravel(),weights=(border&~allowed_top).ravel(),minlength=n)
    top=np.bincount(labels.ravel(),weights=(border&allowed_top).ravel(),minlength=n)
    selected=np.zeros(n,bool);notes=[]
    for i in range(1,n):
        if not overlap[i]:continue
        x,y,w,h,area=map(int,stats[i])
        accepted=(overlap[i]>=32 and 128<=area<=.01*H*W and max(w,h)<=.2*min(H,W) and other[i]==0)
        selected[i]=accepted
        notes.append(dict(id=i,xywh=[x,y,w,h],area=area,core_overlap=int(overlap[i]),
                          top_contacts=int(top[i]),other_contacts=int(other[i]),accepted=bool(accepted)))
    return selected[labels]&connected,notes


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(Path(prior.__file__).relative_to(ROOT)):sha(Path(prior.__file__))}
    for name in ['manifest.json','complete.json']:dependencies[str((prior.OUT/name).relative_to(ROOT))]=sha(prior.OUT/name)
    descriptors=prior.descriptors()
    for r in descriptors:
        source=r['source'];page=r['page'];dependencies[page['original']]=sha(ROOT/page['original'])
        dependencies[str((source/'baseline_review.json').relative_to(ROOT))]=sha(source/'baseline_review.json')
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');pages=[];regions=[]
    for i,r in enumerate(descriptors,1):
        page=r['page'];key=page['page'];base=r['edit'];core=r['core'];domain=r['domain'];guard=r['guard']
        original=read(ROOT/page['original']);source=r['source'];values={};notes={}
        if core.any():
            v6={k:read(source/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
            features,_=prior.anchor.prior.mask.prior.MARGIN.load_features(key,original,v6)
            for level in LEVELS:
                support,detail=candidate(original,features,core,domain,guard,level)
                values[level]=support&~base;notes[level]=detail
            del features,v6
        else:
            for level in LEVELS:values[level]=np.zeros_like(base);notes[level]=[]
        dest=OUT/'pages'/key
        for name,item in dict(base_edit=base,core=core,domain=domain,guard=guard).items():write(dest/(name+'.png'),item)
        for level in LEVELS:
            assert not np.any(values[level]&(guard|~domain))
            for name,item in dict(edit=base|values[level],added=values[level]).items():write(dest/f'drop{level}'/(name+'.png'),item)
        pages.append(dict(page=key,additions={str(k):int(v.sum()) for k,v in values.items()},components={str(k):v for k,v in notes.items()}))
        rois=[dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+page['rois']
        for roi in rois:
            rid=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s];shows=[]
            before=crop.copy();before[base[s]]=(0,200,255)
            for level in LEVELS:
                show=before.copy();show[values[level][s]]=(255,120,0);shows.append(show)
            figure([crop,before]+shows,['原画像','黄＝前回の編集','青＝差6まで連結','青＝差4まで連結'],
                OUT/'comparisons'/(rid+'.png'),roi['title']+' · 受理コアに接続する淡い支持',
                '全ページ判定。固定コアに接続する成分だけ。青は今回の追加。復元なし。',
                '上端16px・本文・左右端・元の印影接続保護を維持。損傷の正解を認定しない。')
            regions.append(dict(id=rid,page=key,xywh=roi['xywh'],additions={str(k):int(v[s].sum()) for k,v in values.items()}))
        print('Anchored pale masks:',i,'/25',key,pages[-1]['additions'],flush=True)
        del original,values,notes;gc.collect()
    for p,d in dependencies.items():assert sha(ROOT/p)==d
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=regions))
    dump(OUT/'verification.json',dict(status='passed',pages=25,domain_and_guards_preserved=True,
        original_edits_retained=True,recursive_chaining=False,ROI_gating=False,restoration=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.py'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete:two pale-support masks on25 development pages',flush=True)


if __name__=='__main__':main()
