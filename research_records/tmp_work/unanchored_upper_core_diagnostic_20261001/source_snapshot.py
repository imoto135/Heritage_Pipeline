"""Original full-domain strong brown components missed by old-band anchoring."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_anchored_pale_margin_masks as prior
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/unanchored_upper_core_diagnostic_20261001'


def infer(original,features,core,domain,guard):
    H,W=domain.shape
    lab=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    support=(features['light_drop']>=10)&(lab[:,:,0]<=80)&domain&~guard
    support=cv2.morphologyEx(support.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    support&=domain&~guard
    n,labels,stats,_=cv2.connectedComponentsWithStats(support.astype(np.uint8),connectivity=8)
    seeds=np.bincount(labels.ravel(),weights=features['color_seed'].ravel(),minlength=n)
    overlap=np.bincount(labels.ravel(),weights=core.ravel(),minlength=n)
    dt=cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,5)
    peak=np.zeros(n,np.float32);np.maximum.at(peak,labels.ravel(),dt.ravel())
    inner=cv2.erode(domain.astype(np.uint8),np.ones((3,3),np.uint8))>0
    allowed_top=np.zeros_like(domain)
    if domain.any():
        yy,xx=np.nonzero(domain);allowed_top[int(yy.min()),int(xx.min())+1:int(xx.max())]=True
    other=np.bincount(labels.ravel(),weights=(support&~inner&~allowed_top).ravel(),minlength=n)
    top=np.bincount(labels.ravel(),weights=(support&~inner&allowed_top).ravel(),minlength=n)
    valid=np.zeros(n,bool);notes=[]
    for i in range(1,n):
        if not seeds[i]:continue
        x,y,w,h,area=map(int,stats[i])
        accepted=128<=area<=.01*H*W and seeds[i]>=32 and peak[i]>=6 and max(w,h)<=.2*min(H,W) and other[i]==0
        valid[i]=accepted
        notes.append(dict(id=i,xywh=[x,y,w,h],area=area,seeds=int(seeds[i]),peak=float(peak[i]),
                          old_core_overlap=int(overlap[i]),top_contacts=int(top[i]),other_contacts=int(other[i]),accepted=bool(accepted)))
    return valid[labels]&support,notes


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(Path(prior.__file__).relative_to(ROOT)):sha(Path(prior.__file__))}
    for name in ['manifest.json','complete.json']:dependencies[str((prior.OUT/name).relative_to(ROOT))]=sha(prior.OUT/name)
    inputs=prior.prior.descriptors()
    for r in inputs:dependencies[r['page']['original']]=sha(ROOT/r['page']['original'])
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=dependencies,pages=25,
        domain='same16px..body-rule minus guards',strong_core='same drop10/L80, area128, brownseed32, peak6, size gates',
        difference='seed support in full safe upper domain rather than old band; top-only contact allowed',
        mask_changes=False,restoration=False,ROI_gating=False,pixel_GT=False)
    dump(OUT/'manifest.json',manifest);shutil.copyfile(__file__,OUT/'source_snapshot.py');pages=[]
    for i,r in enumerate(inputs,1):
        page=r['page'];key=page['page'];original=read(ROOT/page['original']);source=r['source']
        v6={k:read(source/'v6_masks'/key/(k+'.png'),True) for k in ['edit','protect','hold','red','extra']}
        features,_=prior.prior.anchor.prior.mask.prior.MARGIN.load_features(key,original,v6)
        core=r['core'];domain=r['domain'];guard=r['guard'];full_core,notes=infer(original,features,core,domain,guard)
        frozen=read(prior.OUT/'pages'/key/'drop6/edit.png',True)
        novel=full_core&~core;dest=OUT/'pages'/key
        for name,item in dict(full_core=full_core,novel_core=novel,frozen_core=core,frozen_edit=frozen,domain=domain,guard=guard).items():write(dest/(name+'.png'),item)
        dump(dest/'components.json',notes)
        pages.append(dict(page=key,new_core=int(novel.sum()),unedited_new_core=int((novel&~frozen).sum()),
                          accepted_components=[n for n in notes if n['accepted']]))
        regions=[dict(id=key+'_whole',title='ページ全体',xywh=[0,0,original.shape[1],original.shape[0]])]+page['rois']
        for roi in regions:
            rid=roi['id'];x,y,w,h=roi['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s]
            before=crop.copy();before[frozen[s]]=(0,200,255);after=before.copy();after[(novel&~frozen)[s]]=(255,120,0)
            figure([crop,before,after],['原画像','黄＝差6まで固定編集','青＝追加強コアの診断'],OUT/'comparisons'/(rid+'.png'),
                roi['title']+' · 旧帯より上から始まる強コア',
                '暗さ差10、茶色種32、半径6、従来のサイズ条件。青は診断のみで復元しない。',
                '余白の全安全領域で支持成分を調べる。本文・印影・長線の保護は固定。')
        print('Upper core diagnostic:',i,'/25',key,pages[-1]['unedited_new_core'],flush=True)
        del original,v6,features,full_core,novel;gc.collect()
    for p,d in dependencies.items():assert sha(ROOT/p)==d
    dump(OUT/'diagnostics.json',pages)
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.py'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete:25-page upper strong-core diagnostic',flush=True)


if __name__=='__main__':main()
