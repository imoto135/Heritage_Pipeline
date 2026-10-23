"""Isolated matched pilot: replace training fade inputs with source-targeted fading.

Existing labels, non-fade inputs, validation inputs and the frozen trainer stay fixed.
This is an engineering probe with provisional labels, not a formal benchmark or DKDS.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'tmp_work/ragan_paper_aligned_comparison_20261021'
STRESS=ROOT/'tmp_work/makura_damage_challenge_20261021'
OUT=ROOT/'tmp_work/targeted_fade_training_20261021'
RUNNER=ROOT/'tmp_work/makura_supervised_smoke_20261021/code_snapshot/run_makura_supervised_pilot.py'
SEED=20261021

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def rgb(p):return np.asarray(Image.open(p).convert('RGB'))
def gray(a):return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)

def anchors(source,target,valid,patch_id):
    g=gray(source)
    ink=(target==0)&(valid>0)&(g<180)
    thin=skeletonize(ink)&(distance_transform_edt(ink)<=2.5)
    thin[:80]=False;thin[432:]=False;thin[:,:80]=False;thin[:,432:]=False
    candidates=np.argwhere(thin)
    rng=np.random.default_rng(int(hashlib.sha256((str(SEED)+patch_id).encode()).hexdigest()[:16],16))
    order=rng.permutation(len(candidates));chosen=[]
    for idx in order:
        y,x=map(int,candidates[idx])
        if all((y-yy)**2+(x-xx)**2>=100**2 for yy,xx in chosen):chosen.append((y,x))
        if len(chosen)==3:break
    return chosen

def build_input(source,alpha,centers,strength):
    field=np.exp(-.5*(distance_transform_edt(~centers)/12.)**2).astype(np.float32)
    field[field<.005]=0
    paper=cv2.morphologyEx(source,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    weight=(field*strength)[...,None]
    out=np.rint(np.clip(source.astype(np.float32)*(1-weight)+paper.astype(np.float32)*weight,0,255)).astype(np.uint8)
    return out,field,paper

def prepare():
    assert not OUT.exists(),f'Preserve existing experiment: {OUT}'
    OUT.mkdir(parents=True)
    annotation=STRESS/'thin_annotations_source_only.json'
    marked={p:np.asarray(Image.open(STRESS/'source_thin_review'/(p+'_high_centers.png')))>0
        for p in ('200017197_00103_1_p00','200017197_00214_1_p00')}
    original=[json.loads(x) for x in (OLD/'unet_v3/train_manifest.jsonl').read_text().splitlines()]
    revised=[];checks=[];cache={}
    for row in original:
        r=dict(row)
        if row['damage_family']=='fade':
            patch=row['patch_id']
            if patch not in cache:
                source=rgb(ROOT/row['clean']);label=row['labels']
                target=np.asarray(Image.open(ROOT/label['target']).convert('L'))
                valid=np.asarray(Image.open(ROOT/label['valid']).convert('L'))
                selected=anchors(source,target,valid,patch)
                centers=np.zeros((512,512),bool)
                for y,x in selected:centers[y,x]=True
                traced=marked.get(patch)
                if traced is not None:centers|=traced
                assert centers.any(),patch
                cache[patch]=(source,centers,selected,traced)
            source,centers,selected,traced=cache[patch]
            strength=float(row['damage_strength'])
            out,field,paper=build_input(source,None,centers,strength)
            dest=OUT/'inputs'/patch/(f"fade_{strength:.2f}.png");dest.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(out).save(dest)
            r['input']=str(dest.relative_to(ROOT));r['input_sha256']=sha(dest)
            r['damage_family']='targeted_fade'
            sy,sx=np.where(centers);delta=gray(out).astype(np.int16)-gray(source).astype(np.int16)
            trace_stats=None
            if traced is not None:
                ty,tx=np.where(traced)
                trace_stats=dict(n=len(tx),changed_ge10=int((np.abs(delta[ty,tx])>=10).sum()),
                    median_delta=float(np.median(delta[ty,tx])),
                    median_remaining_contrast=float(np.median(gray(paper)[ty,tx].astype(np.int16)-gray(out)[ty,tx].astype(np.int16))))
            checks.append(dict(patch=patch,strength=strength,source_sha256=sha(ROOT/row['clean']),
                input_sha256=r['input_sha256'],selected_generic_xy=[[x,y] for y,x in selected],
                center_n=len(sx),center_changed_ge10=int((np.abs(delta[sy,sx])>=10).sum()),
                center_median_delta=float(np.median(delta[sy,sx])),trace=trace_stats))
        revised.append(r)
    assert len(revised)==864 and len(checks)==288 and len(cache)==96
    assert all(a['id']==b['id'] for a,b in zip(original,revised))
    assert all(a['labels']==b['labels'] for a,b in zip(original,revised))
    assert all(a['input_sha256']==b['input_sha256'] for a,b in zip(original,revised) if a['damage_family']!='fade')
    for model,condition in [('unet','unet_v3'),('ragan','ragan_public_v3')]:
        old=OLD/condition;new=OUT/condition
        new.mkdir()
        (new/'train_manifest.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in revised))
        for name in ('val_manifest.jsonl','label_protocol.json','adapter_verification.json','code_hashes.json'):
            shutil.copy2(old/name,new/name)
        dump(new/'condition.json',dict(model=model,source_condition=condition,
            difference='training fade RGB inputs only; 288 of 864 rows',same_targets=True,
            same_validation=True,same_seed=True,formal_accuracy=False))
    dump(OUT/'protocol.json',dict(purpose='Matched source-targeted fade engineering pilot',
        source_condition=['unet_v3','ragan_public_v3'],changed_train_rows=288,total_train_rows=864,
        validation='fixed v2 validation manifest unchanged',labels='v3 provisional training labels unchanged',
        fade='Gaussian sigma12 native px on three thin skeleton anchors per patch; include source-traced high centers in two train patches; blend source RGB to morphology-close41 paper estimate',
        strengths=[.30,.55,.80],seed=SEED,steps=432,batch_size=4,
        annotation_sha256=sha(annotation),source_trace_used_for_training_only=True,
        source_trace_from_validation_used=False,developer_known_cases=True,formal_accuracy=False,
        dkds_reproduction=False))
    dump(OUT/'input_audit.json',dict(rows=checks))
    # Source and changed-input examples; never use validation trace in training.
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    for patch in ('200017197_00103_1_p00','200017197_00214_1_p00'):
        source,_,_,_=cache[patch]
        inputs=[source]+[rgb(OUT/'inputs'/patch/(f'fade_{s:.2f}.png')) for s in (.3,.55,.8)]
        canvas=Image.new('RGB',(4*512+5*12,512+42+24),'#faf9f5');draw=ImageDraw.Draw(canvas)
        for i,(name,im) in enumerate(zip(('原本','退色 .30','退色 .55','退色 .80'),inputs)):
            x=12+i*(512+12);draw.text((x,8),name,font=font,fill='#203442');canvas.paste(Image.fromarray(im),(x,42))
        path=OUT/'figures'/(patch+'.png');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    print('Prepared',len(checks),'targeted fade inputs across',len(cache),'train patches',flush=True)
    for c in checks:
        if c['trace']:print(c['patch'],c['strength'],c['trace'],flush=True)

def run(condition,device):
    import ragan_pilot_support as support
    support.ROOT=ROOT;support.REFERENCE=ROOT/'tmp_work/ra_gan_reference_20261021'
    spec=importlib.util.spec_from_file_location('fixed_pilot_targeted',RUNNER)
    runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
    runner.ROOT=ROOT;runner.OUT=OUT/condition
    model=json.loads((runner.OUT/'condition.json').read_text())['model']
    args=argparse.Namespace(model=model,steps=432,batch_size=4,lr=3e-5,validate_every=108,device=device)
    runner.train(args)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--condition',choices=['unet_v3','ragan_public_v3'])
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        assert a.condition
        run(a.condition,a.device)
