"""Whole-page geometry and retained-paper orientation, diagnostic only."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import prepare_chromatic_external_baselines as references
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=references.ROOT
OUT=ROOT/'tmp_work/directional_paper_candidate_diagnostic_20261001'
SIGMAS=[0,1,2]
SETTINGS=dict(pages=25,ROIs=77,baseline='frozen upper strong + horizontal exclusion',
    target='whole-page eligible edit components, core overlap>=32 and area>=64',
    shape='coordinate covariance major axis and anisotropy',
    references='qualified known paper and separately retained unedited safe pixels',
    ring_px=64,minimum_gradient_samples=256,
    gradient='Gaussian sigma0/1/2 with explicit6sigma+1 kernel, Sobel3 /8, clip +/-5',
    provenance='erode reference by6sigma+3, so every gradient stencil pixel is observed reference',
    orientation='minor eigenvector of mean gradient outer product, no gradient mean subtraction',
    provisional_match=dict(sigma=1,reference='retained unedited safe pixels',anisotropy=.55,
        shape_anisotropy=.8,alignment_degrees=15,gradient_energy=.05,
        area_max_page_fraction=.01,max_side_short_page_fraction=.2),
    inference_before_ROIs=True,masks_changed=False,restoration_changed=False,pixel_GT=False,new_DDRM=False,development=True)


def direction(x,y):
    tensor=np.array([[np.mean(x*x),np.mean(x*y)],[np.mean(x*y),np.mean(y*y)]],np.float64)
    vals,vec=np.linalg.eigh(tensor);energy=float(vals.sum())
    return dict(tensor=tensor.tolist(),eigenvalues=vals.tolist(),gradient_energy=energy,
        anisotropy=float((vals[1]-vals[0])/max(energy,1e-6)),
        orientation_degrees=float(np.degrees(np.arctan2(vec[1,0],vec[0,0]))%180))


def infer(light,known,eligible,core,domain,guard):
    H,W=eligible.shape;retained=domain&~guard&~eligible
    # Eligible already excludes guarded edit, so also remove every frozen edit via supplied known mask below.
    spectra={};valid={}
    for sigma in SIGMAS:
        sm=cv2.GaussianBlur(light,(6*sigma+1,6*sigma+1),sigma) if sigma else light
        gx=np.clip(cv2.Sobel(sm,cv2.CV_32F,1,0,ksize=3,scale=1/8),-5,5)
        gy=np.clip(cv2.Sobel(sm,cv2.CV_32F,0,1,ksize=3,scale=1/8),-5,5)
        spectra[sigma]=(gx,gy)
        for name,reference in [('known',known&retained),('retained',retained)]:
            valid[name,sigma]=cv2.erode(reference.astype(np.uint8),np.ones((6*sigma+3,6*sigma+3),np.uint8),
                borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    n,labels,stats,_=cv2.connectedComponentsWithStats(eligible.astype(np.uint8),connectivity=4)
    overlap=np.bincount(labels.ravel(),weights=core.ravel(),minlength=n);notes=[];matches=np.zeros_like(eligible)
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<64 or overlap[k]<32:continue
        x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
        local=labels[y0:y1,x0:x1]==k;yy,xx=np.nonzero(local)
        covariance=np.cov(np.stack([xx,yy]),bias=True);vals,vec=np.linalg.eigh(covariance)
        shape=dict(covariance=covariance.tolist(),anisotropy=float((vals[1]-vals[0])/max(vals.sum(),1e-6)),
            orientation_degrees=float(np.degrees(np.arctan2(vec[1,1],vec[0,1]))%180),
            center_xy=[float(xx.mean()+x0),float(yy.mean()+y0)])
        near=cv2.distanceTransform((~local).astype(np.uint8),cv2.DIST_L2,5)<=64;details={}
        for name in ['known','retained']:
            details[name]={}
            for sigma in SIGMAS:
                support=valid[name,sigma][y0:y1,x0:x1]&near
                metric=dict(samples=int(support.sum()),available=bool(support.sum()>=256))
                if metric['available']:
                    gx,gy=spectra[sigma];metric.update(direction(gx[y0:y1,x0:x1][support],gy[y0:y1,x0:x1][support]))
                    difference=abs(metric['orientation_degrees']-shape['orientation_degrees'])
                    metric['alignment_degrees']=float(min(difference,180-difference))
                details[name][str(sigma)]=metric
        d=details['retained']['1'];shape_ok=area<=.01*H*W and max(w,h)<=.2*min(H,W)
        matched=bool(d['available'] and d['anisotropy']>=.55 and d['gradient_energy']>=.05 and shape['anisotropy']>=.8
                     and d['alignment_degrees']<=15 and shape_ok)
        if matched:matches[y0:y1,x0:x1]|=local
        notes.append(dict(id=k,xywh=[x,y,w,h],area=area,core_overlap=int(overlap[k]),shape=shape,
            direction=details,size_ok=shape_ok,provisional_matched=matched,
            meaning='orientation match only; physical cause and whether editing is correct are unknown'))
    return labels,notes,matches,retained


def main():
    OUT.mkdir(parents=True,exist_ok=True);prior=references.prior
    check_bundle(prior.mask.OUT);inputs=prior.mask.pale.prior.descriptors()
    rows=[]
    for r in json.loads((prior.OUT/'diagnostics.json').read_text()):rows.append({**r,'baseline_local':str((prior.OUT/'native'/r['id']/'short_upper').relative_to(ROOT))})
    rows.extend(json.loads((references.OUT/'diagnostics.json').read_text()));assert len(rows)==77
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(Path(references.__file__).relative_to(ROOT)):sha(Path(references.__file__))}
    for folder in [prior.mask.OUT,prior.OUT,references.OUT]:
        for name in ['manifest.json','complete.json']:deps[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for desc in inputs:
        key=desc['page']['page'];edit=read(prior.mask.OUT/'pages'/key/'short_upper/edit.png',True)
        _,path=references.full_reference(desc,edit);deps[str(path.relative_to(ROOT))]=sha(path);deps[desc['page']['original']]=sha(ROOT/desc['page']['original'])
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');pages=[];regions=[]
    for index,desc in enumerate(inputs,1):
        key=desc['page']['page'];original=read(ROOT/desc['page']['original']);folder=prior.mask.OUT/'pages'/key;dest=OUT/'pages'/key
        edit=read(folder/'short_upper/edit.png',True);domain=read(folder/'domain.png',True);guard=read(folder/'guard.png',True)
        core=read(folder/'core.png',True)|read(folder/'short_upper/strong_core.png',True)
        known,_=references.full_reference(desc,edit);eligible=edit&domain&~guard
        light=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        labels,notes,matched,retained=infer(light,known,eligible,core,domain,guard)
        assert not np.any(retained&edit)
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'labels.npz',labels=labels)
        for name,item in dict(edit=edit,eligible=eligible,core=core,domain=domain,guard=guard,known=known,retained=retained,provisional_matches=matched).items():write(dest/(name+'.png'),item)
        dump(dest/'components.json',notes);pages.append(dict(page=key,provisional_pixels=int(matched.sum()),matched_components=[n for n in notes if n['provisional_matched']],components=len(notes)))
        for row in [r for r in rows if r['page']==key]:
            rid=row['id'];x,y,w,h=row['xywh'];s=(slice(y,y+h),slice(x,x+w));crop=original[s];base=read(ROOT/row['baseline_local']/'output.png')
            relevant=[n for n in notes if np.any(labels[s]==n['id'])];panels=[]
            for scope,color in [('known',(255,120,0)),('retained',(210,30,200))]:
                show=crop.copy()
                for note in relevant:
                    center=np.rint(np.array(note['shape']['center_xy'])-[x,y]).astype(int);length=min(50,max(10,min(note['xywh'][2:])//2))
                    for angle,c in [(note['shape']['orientation_degrees'],(40,210,40))]+([(note['direction'][scope]['1']['orientation_degrees'],color)] if note['direction'][scope]['1']['available'] else []):
                        delta=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])*length
                        start=tuple(np.rint(center-delta).astype(int));end=tuple(np.rint(center+delta).astype(int));cv2.arrowedLine(show,start,end,c,2,tipLength=.18)
                panels.append(show)
            elsewhere='旧14ページ固定ROIの紙面外は保存DDRM。' if row['base_elsewhere']=='saved_DDRM' else '紙面外は原画像。'
            figure([crop,base]+panels,['原画像','固定マスクの補修基準','緑＝形状、青＝資格紙面σ1','緑＝形状、紫＝保持画素σ1'],OUT/'comparisons'/(rid+'.png'),
                row['title']+' · 形状と周囲の方向の診断','周囲64px・元画素だけ。保持画素には紙面資格外もあり、模様の復元源にはしない。',
                elsewhere+' 方向が似ても物理原因を断定しない。診断のみ。')
            regions.append(dict(id=rid,page=key,xywh=row['xywh'],provisional_pixels=int(matched[s].sum()),components=relevant))
        print('Directional paper diagnostic:',index,'/25',key,'matches',int(matched.sum()),flush=True)
        del original,light,labels,known;gc.collect()
    for path,value in deps.items():assert sha(ROOT/path)==value
    dump(OUT/'diagnostics.json',dict(pages=pages,regions=regions))
    dump(OUT/'verification.json',dict(status='passed',whole_page_inference=True,masks_and_output_changes=0,pages=25,ROIs=77,gradient_stencils_observed_only=True,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:77 PNGs for whole-page directional diagnostic only',flush=True)


if __name__=='__main__':main()
