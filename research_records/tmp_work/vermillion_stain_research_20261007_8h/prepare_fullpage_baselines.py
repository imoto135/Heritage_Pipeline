"""Native pages, fixed Sauvola black and independently compared red selectors."""
from pathlib import Path
import json
import importlib.util
import sys
import time
import os
import hashlib
import cv2
import numpy as np
from PIL import Image
from synthetic_layers import OUT, ROOT, local_paper, file_hash

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def sauvola(rgb):
    g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    m=cv2.boxFilter(g,-1,(61,61),borderType=cv2.BORDER_REFLECT_101)
    var=np.maximum(cv2.boxFilter(g*g,-1,(61,61),borderType=cv2.BORDER_REFLECT_101)-m*m,0)
    return g<m*(1+.2*(np.sqrt(var)/.5-1))

def ridge(field):
    result=np.zeros(field.shape,np.float32)
    for sigma in [1.,2.,4.,8.]:
        a=cv2.GaussianBlur(field,(0,0),sigma,borderType=cv2.BORDER_REFLECT)
        xx=cv2.Sobel(a,cv2.CV_32F,2,0,ksize=3,scale=sigma*sigma*.25)
        yy=cv2.Sobel(a,cv2.CV_32F,0,2,ksize=3,scale=sigma*sigma*.25)
        xy=cv2.Sobel(a,cv2.CV_32F,1,1,ksize=3,scale=sigma*sigma*.25)
        dd=np.sqrt((xx-yy)**2+4*xy*xy)
        eig1=.5*(xx+yy-dd);eig2=.5*(xx+yy+dd)
        large=np.where(np.abs(eig1)>np.abs(eig2),eig1,eig2);small=np.where(np.abs(eig1)>np.abs(eig2),eig2,eig1)
        rr=np.maximum(-large,0)*np.exp(-2*(small/(np.abs(large)+1e-8))**2)
        rr[np.abs(small)>.5*np.abs(large)]=0
        result=np.maximum(result,rr)
    return result

def heuristic_masks(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);hh,ss,vv=np.moveaxis(hsv,2,0)
    strict=((hh<=12)|(hh>=168))&(ss>180)
    loose=((hh<=25)|(hh>=168))&(ss>=30)&(vv>=30)
    bg=local_paper(rgb);od=np.log((bg+1)/(rgb.astype(np.float32)+1))
    green=od[:,:,1]-od[:,:,0];blue=od[:,:,2]-od[:,:,0]
    relative=(green>=.012)&(blue>=.012)&(rgb[:,:,0]>=rgb[:,:,1])
    response=ridge(np.minimum(green,blue))
    seeds=relative&(response>=.004)
    neighbourhood=cv2.dilate(seeds.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    return dict(hsv_strict=strict,hsv_loose=loose,relative_chroma=relative,relative_ridge=relative&neighbourhood)

def main():
    cv2.setNumThreads(2)
    v1_path=ROOT/'modules/ddrm_npj/scripts/csc.py';ace_path=ROOT/'modules/ddrm_npj/scripts/ace.py'
    v2_path=ROOT/'tmp_work/npj_dirt_control_20261002/source_snapshot/ddrm_npj_v2/scripts/csc.py'
    v1=module('vermilion_local_csc_v1',v1_path);v2=module('vermilion_local_csc_v2',v2_path);ace=module('vermilion_ace',ace_path)
    dest=OUT/'fullpage_baselines';dest.mkdir(exist_ok=True)
    protocol=json.loads((OUT/'cases_frozen.json').read_text())
    pages={r['page']:r for r in protocol['cases']}
    settings=dict(native_resolution=True,mask_storage='white=positive class; pipeline separately uses black=target',
        sauvola=dict(window=61,k=.2,R=.5,border='REFLECT_101'),
        CSC='Local v1/v2, not verified authentic paper implementation',
        ace=dict(alpha=5.,n_samples=200,seed=0),csc=dict(seed=42,max_samples=50000,border_margin_ratio=.04),
        thresholds='Fixed exploratory heuristics before new outputs; no novelty claim for ridge/color combination',
        method_settings=dict(hsv_strict='h<=12 or>=168,S>180',hsv_loose='h<=25 or>=168,S>=30,V>=30',
            relative_chroma='input-estimated OD G-R>=.012,B-R>=.012,R>=G',
            relative_ridge='relative_chroma & within2px of Hessian ridge(min(dG,dB))>=.004; sigma1,2,4,8'),
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in [v1_path,v2_path,ace_path,Path(__file__)]},
        cases_sha256=file_hash(OUT/'cases_frozen.json'))
    (dest/'protocol.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2))
    summary=[]
    for page,r in pages.items():
        pp=dest/page;pp.mkdir(exist_ok=True);start=time.monotonic()
        source=ROOT/r['original'];assert file_hash(source)==r['source_sha256']
        rgb=np.asarray(Image.open(source).convert('RGB'));bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        # Reuse only input-identical ACE from the previous development collection.
        prior=ROOT/'tmp_work/red_positive_fixed_check_20260930/baseline'/page
        prior_done=prior/'ace_complete.json'
        if prior_done.exists() and json.loads(prior_done.read_text())['original_sha256']==r['source_sha256']:
            record=json.loads(prior_done.read_text());assert file_hash(prior/'ace.png')==record['sha256']
            equalized=cv2.imread(str(prior/'ace.png'));ace_origin='verified prior ACE'
        else:
            equalized=ace.automatic_color_equalization(bgr,alpha=5.,n_samples=200,seed=0);ace_origin='computed frozen ACE'
        cv2.imwrite(str(pp/'ace.png'),equalized)
        masks=heuristic_masks(rgb);masks['sauvola_black']=sauvola(rgb)
        for name,mod in [('csc_v1',v1),('csc_v2',v2)]:
            c=mod.color_space_clustering(equalized,seed=42,max_samples=50000,border_margin_ratio=.04)
            for kind,field in [('red',c.red_text_mask),('black',c.black_text_mask),('stain',c.degradation_mask)]:
                masks[name+'_'+kind]=field>127
        for name,mask in masks.items():
            assert mask.shape==rgb.shape[:2];Image.fromarray(mask.astype(np.uint8)*255).save(pp/(name+'.png'))
        record=dict(page=page,book=r['book'],original=r['original'],source_sha256=r['source_sha256'],
            seconds=time.monotonic()-start,ACE_origin=ace_origin,counts={k:int(v.sum()) for k,v in masks.items()},
            outputs={p.name:file_hash(p) for p in pp.glob('*.png')})
        (pp/'complete.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));summary.append(record)
        (dest/'progress.json').write_text(json.dumps(dict(completed=len(summary),total=len(pages),latest=page)))
        print('baseline',page,round(record['seconds'],1),'sec; v2 red',record['counts']['csc_v2_red'],flush=True)
    (dest/'complete.json').write_text(json.dumps(dict(status='completed',pages=summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
