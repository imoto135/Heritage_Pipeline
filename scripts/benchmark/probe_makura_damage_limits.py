"""Frozen development stress test; provisional labels are not expert GT.

More difficult damage is a calibration experiment, never evidence by itself
that deep learning can recover information no longer present in the input.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from skimage.morphology import skeletonize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tmp_work/makura_supervised_pilot_20261021'
LABELS=ROOT/'tmp_work/makura_supervised_smoke_20261021'
OUT=ROOT/'tmp_work/makura_damage_challenge_20261021'
FAMILIES=('irregular_stain','carbon_spot','water_rim','thin_fade_grain')
STRENGTHS=(.35,.65,.90)
HOWE={.25:[40.,160.,640.,2560.],.375:[float(np.sqrt(12800)),640.],.4:[160.,640.]}
SAUVOLA=[(31,.2),(61,.1),(61,.2),(61,.3),(121,.2)]
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def dump(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path):return str(path.relative_to(ROOT))
def rgb(path):return np.array(Image.open(path).convert('RGB'))
def gray(path):return np.array(Image.open(path).convert('L'))
def seed(*parts):return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')
def save_mask(path,ink):
    path.parent.mkdir(exist_ok=True,parents=True)
    Image.fromarray(np.where(ink,0,255).astype(np.uint8)).save(path)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def materials(raw,ink,valid,split,patch,family):
    rng=np.random.default_rng(seed('damage-stress-v2',split,patch,family))
    h,w=ink.shape;yy,xx=np.mgrid[:h,:w]
    core=np.zeros_like(ink);core[80:-80,80:-80]=True
    # Candidate skeleton is used for positioning, not independent truth.
    distance=cv2.distanceTransform(ink.astype(np.uint8),cv2.DIST_L2,5)
    thin=skeletonize(ink)&(distance<=1.5)&core
    candidates=thin if family=='thin_fade_grain' and thin.any() else ink&valid&core
    points=np.argwhere(candidates);assert len(points)
    y,x=points[int(rng.integers(len(points)))];anchor=[int(x),int(y)]
    if family=='carbon_spot':
        nearby=cv2.distanceTransform((~ink).astype(np.uint8),cv2.DIST_L2,5)
        points=np.argwhere((nearby>=4)&(nearby<=12)&valid&core)
        y,x=points[int(rng.integers(len(points)))];anchor=[int(x),int(y)]
    rx,ry=(rng.uniform(25,40,2) if family=='carbon_spot' else rng.uniform(65,115,2))
    if family=='water_rim':x=x-rx*.8
    angle=rng.uniform(-np.pi,np.pi);dx=xx-x;dy=yy-y
    u=(dx*np.cos(angle)+dy*np.sin(angle))/rx
    v=(-dx*np.sin(angle)+dy*np.cos(angle))/ry
    theta=np.arctan2(v,u)
    modulation=1+.13*np.sin(3*theta+rng.uniform(0,6))+.08*np.sin(7*theta+rng.uniform(0,6))
    radius=np.sqrt(u*u+v*v)/modulation
    if family=='water_rim':
        field=np.exp(-.5*((radius-1)/.065)**2)+.14*np.clip(1-radius,0,1)
    else:
        edge=.13 if family=='carbon_spot' else .32
        field=np.clip((1-radius)/edge+.5,0,1)
    coarse=cv2.resize(rng.random((17,17)).astype(np.float32),(w,h),interpolation=cv2.INTER_CUBIC)
    coarse=cv2.GaussianBlur(coarse,(0,0),2)
    coarse=(coarse-coarse.min())/max(float(np.ptp(coarse)),1e-8)
    field=field*(.80+.20*coarse);field[field<.01]=0
    field=(field/max(float(field.max()),1e-8)).astype(np.float32)
    noise=rng.normal(0,1,(h,w)).astype(np.float32)
    noise=cv2.GaussianBlur(noise,(0,0),.6);noise/=noise.std()
    return dict(field=field,noise=noise,anchor_xy=anchor,center_xy=[float(x),float(y)],
        radii=[float(rx),float(ry)],seed=seed('damage-stress-v2',split,patch,family))


def apply_damage(raw,paper,mat,family,strength):
    if family=='identity':return raw.copy()
    f=mat['field']*strength;source=raw.astype(np.float32)
    if family=='irregular_stain':
        result=source*np.exp(-f[...,None]*np.array([2.,2.6,3.3],np.float32))
    elif family=='carbon_spot':
        result=source*(1-f[...,None])+np.array([23.,21.,18.],np.float32)*f[...,None]
    elif family=='water_rim':
        result=source*np.exp(-f[...,None]*np.array([1.7,2.7,3.7],np.float32))
    else:
        assert family=='thin_fade_grain'
        result=source*(1-f[...,None])+paper.astype(np.float32)*f[...,None]
        result+=mat['noise'][...,None]*(7*f[...,None])
    return np.rint(np.clip(result,0,255)).astype(np.uint8)


def prepare():
    assert not (OUT/'protocol.json').exists(),'Preserve the frozen experiment.'
    old=json.loads((SOURCE/'protocol.json').read_text())
    pages=[p for split,n in [('train',4),('val',2)] for p in [x for x in old['pages'] if x['split']==split][:n]]
    index={p['patch_id']:p for p in json.loads((SOURCE/'patch_index.json').read_text())}
    labels={p['patch_id']:p for p in json.loads((LABELS/'labels_index.json').read_text())}
    selections=[]
    for page in pages:
        patch=page['page']+'_p00';original=ROOT/index[patch]['folder']/'clean.png';label=labels[patch]
        selections.append(dict(patch_id=patch,page=page['page'],group=page['group'],split=page['split'],
            original=rel(original),original_sha256=sha(original),target=label['target'],
            target_sha256=label['target_sha256'],valid=label['valid'],valid_sha256=label['valid_sha256']))
    protocol=dict(selections=selections,families=FAMILIES,strengths=STRENGTHS,howe=HOWE,sauvola=SAUVOLA,
        csc='Local v2 default; ACE alpha5,200 samples,seed0; JPEG95',
        population='First four train pages and first two val pages, first native patch each; within-book development calibration',
        selected_without_new_results=True,final_test=False,
        target='Frozen before artificial damage; old masked provisional labels, not expert GT',
        thin_proxy='Candidate skeleton radius <=1.5px, scored separately, unreviewed and not independent GT',
        metrics='Masked ink recall and paper false positive rate separately, center 384x384; r0 and r2 ink coverage',
        counterfactual='Same damage applied to original and morphology-close paper estimate; remaining RGB contrast is diagnostic, not human legibility',
        visibility='Near-occlusion/sensor-floor cases remain separate; failure cannot establish recoverability by learning',
        materials='Procedural only; no actual evaluation stain crops reused as learning textures',
        interpretation='Eight fixed Howe settings do not exhaust all classical algorithms or parameters')
    dump(OUT/'protocol.json',protocol)
    sources=[Path(__file__),ROOT/'scripts/benchmark/audit_classical_limits.py',
        ROOT/'tmp_work/howe_diagnosis_20261008/experiment.py',ROOT/'tmp_work/howe_diagnosis_20261008/sweep_cli',
        ROOT/'tmp_work/classical_deep_review_20261008/run.py',
        ROOT/'tmp_work/classical_black_comparison_20261008/compare.py',
        ROOT/'modules/ddrm_npj_v2/scripts/csc.py']
    dump(OUT/'dependencies.json',{rel(p):sha(p) for p in sources})
    for source in sources:
        dest=OUT/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(source,dest)
    print('Frozen',len(selections),'patches, 78 inputs, 14 classical settings',flush=True)


def measure(pred,truth,valid,thin):
    positive=truth&valid;negative=~truth&valid
    dilated=cv2.dilate(pred.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0
    return dict(ink_n=int(positive.sum()),ink_hit=int((pred&positive).sum()),ink_r2=int((dilated&positive).sum()),
        nonink_n=int(negative.sum()),nonink_false=int((pred&negative).sum()),
        thin_candidate_n=int(thin.sum()),thin_candidate_hit=int((pred&thin).sum()),thin_candidate_r2=int((dilated&thin).sum()),
        balanced_error=float((~pred&positive).sum()/max(1,positive.sum())+(pred&negative).sum()/max(1,negative.sum())))


def worker(selection):
    cv2.setNumThreads(1);started=time.monotonic();patch=selection['patch_id']
    for path,digest in json.loads((OUT/'dependencies.json').read_text()).items():assert sha(ROOT/path)==digest,path
    for name in ('original','target','valid'):assert sha(ROOT/selection[name])==selection[name+'_sha256']
    classic=module(ROOT/'scripts/benchmark/audit_classical_limits.py','stress_classic')
    howe=module(ROOT/'tmp_work/howe_diagnosis_20261008/experiment.py','stress_howe')
    csc=module(ROOT/'modules/ddrm_npj_v2/scripts/csc.py','stress_csc')
    csc._filter_red_candidate_by_component=classic.component_filter
    raw=rgb(ROOT/selection['original']);truth=gray(ROOT/selection['target'])==0
    valid=gray(ROOT/selection['valid'])>0;valid[:64]=False;valid[-64:]=False;valid[:,:64]=False;valid[:,-64:]=False
    distance=cv2.distanceTransform(truth.astype(np.uint8),cv2.DIST_L2,5)
    central=np.zeros_like(truth);central[64:-64,64:-64]=True
    thin=skeletonize(truth)&(distance<=1.5)&central
    paper=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((41,41),np.uint8))
    variants=[('identity',0.)]+[(family,strength) for family in FAMILIES for strength in STRENGTHS]
    records=[]
    for family,strength in variants:
        name=f'{family}_{strength:.2f}';folder=OUT/'cases'/patch/name;folder.mkdir(exist_ok=True,parents=True)
        material=materials(raw,truth,valid,selection['split'],patch,family) if family!='identity' else None
        damaged=apply_damage(raw,paper,material,family,strength)
        Image.fromarray(damaged).save(folder/'input.png')
        if family=='identity':
            assert np.array_equal(damaged,raw);changed=np.zeros(truth.shape,bool);visibility=None
        else:
            repeat=apply_damage(raw,paper,materials(raw,truth,valid,selection['split'],patch,family),family,strength)
            assert np.array_equal(damaged,repeat)
            changed=np.max(abs(damaged.astype(np.int16)-raw.astype(np.int16)),axis=2)>=2
            assert (changed&truth&valid).any(),(patch,family,'damage missed known ink')
            counter=apply_damage(paper,paper,material,family,strength)
            signal=np.max(abs(damaged.astype(np.int16)-counter.astype(np.int16)),axis=2)
            affected=(material['field']>.1)&truth&valid
            assert affected.any()
            visibility=dict(affected_known_ink=int(affected.sum()),median_rgb_contrast=float(np.median(signal[affected])),
                fraction_contrast_below2=float((signal[affected]<2).mean()),fraction_contrast_below8=float((signal[affected]<8).mean()),
                warning='Paper is a morphology estimate. These values do not establish human readability.')
            Image.fromarray(np.rint(material['field']*strength*255).astype(np.uint8)).save(folder/'damage_field.png')
        g=howe.deep.base.gray(damaged);scores=[]
        def record(key,mask,parameters):
            assert mask.shape==truth.shape and mask.dtype==bool
            save_mask(folder/(key+'.png'),mask)
            scores.append(dict(method=key,parameters=parameters,sha256=sha(folder/(key+'.png')),**measure(mask,truth,valid,thin)))
        for window,k in SAUVOLA:
            record(f'sauvola_w{window}_k{k:.2f}',howe.deep.base.sauvola(g,window,k),dict(window=window,k=k))
        for threshold,weights in HOWE.items():
            terms,_=howe.terms(g,thi=threshold)
            for pre,meta in howe.sweep(terms,weights,OUT/'scratch'/patch,name+f'_t{threshold}'):
                mask,_=howe.post(pre)
                record(f'howe_t{threshold:.3f}_c{meta["c"]:.3f}',mask,dict(thi=threshold,c=meta['c'],canny='local skimage approximation'))
        ace=classic.ace_chunked(cv2.cvtColor(damaged,cv2.COLOR_RGB2BGR))
        ok,encoded=cv2.imencode('.jpg',ace,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
        ace=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
        with threadpool_limits(limits=1):result=csc.color_space_clustering(ace)
        record('csc_v2',result.black_text_mask>0,dict(ace_jpeg_quality=95))
        record_info=dict(patch_id=patch,split=selection['split'],family=family,strength=strength,folder=rel(folder),
            input_sha256=sha(folder/'input.png'),target_sha256=selection['target_sha256'],
            valid_sha256=selection['valid_sha256'],changed_known_ink=int((changed&truth&valid).sum()),
            material={k:v for k,v in material.items() if k not in ('field','noise')} if material else None,
            visibility=visibility,scores=scores,formal_gt=False)
        dump(folder/'result.json',record_info);records.append(record_info)
        print(patch,name,'done',round(time.monotonic()-started,1),flush=True)
    dump(OUT/'results'/(patch+'.json'),records)
    return patch


def run(workers):
    protocol=json.loads((OUT/'protocol.json').read_text())
    pending=[p for p in protocol['selections'] if not (OUT/'results'/(p['patch_id']+'.json')).exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(worker,p) for p in pending]):f.result()
    records=[r for p in protocol['selections'] for r in json.loads((OUT/'results'/(p['patch_id']+'.json')).read_text())]
    dump(OUT/'all_results.json',records);print('Completed',len(records),'inputs',flush=True)


def finish():
    protocol=json.loads((OUT/'protocol.json').read_text());records=json.loads((OUT/'all_results.json').read_text())
    selections={p['patch_id']:p for p in protocol['selections']}
    assert len(records)==78 and sum(len(r['scores']) for r in records)==1092
    for selection in selections.values():
        for name in ('original','target','valid'):assert sha(ROOT/selection[name])==selection[name+'_sha256']
    for r in records:
        folder=ROOT/r['folder'];assert sha(folder/'input.png')==r['input_sha256']
        for score in r['scores']:assert sha(folder/(score['method']+'.png'))==score['sha256']
    groups={s:{p['group'] for p in selections.values() if p['split']==s} for s in ('train','val')}
    assert not groups['train']&groups['val']
    figures=[];selected=[]
    first_val=next(p['patch_id'] for p in protocol['selections'] if p['split']=='val')
    for family in FAMILIES:
        selected.append((f'fixed_val_{family}',next(r for r in records if r['patch_id']==first_val and r['family']==family and r['strength']==.65),False))
        candidates=[r for r in records if r['split']=='train' and r['family']==family]
        # Development discovery using all-eight-config Howe lower envelope.
        chosen=max(candidates,key=lambda r:min(s['balanced_error'] for s in r['scores'] if s['method'].startswith('howe_')))
        selected.append((f'exploratory_train_{family}',chosen,True))
    for identifier,r,exploratory in selected:
        folder=ROOT/r['folder'];source=selections[r['patch_id']]
        original=rgb(ROOT/source['original']);target=rgb(ROOT/source['target'])
        target[gray(ROOT/source['valid'])==0]=[235,40,150]
        oracle=min([s for s in r['scores'] if s['method'].startswith('howe_')],key=lambda s:s['balanced_error'])
        best_s=min([s for s in r['scores'] if s['method'].startswith('sauvola_')],key=lambda s:s['balanced_error'])
        keys=['csc_v2','sauvola_w61_k0.20','howe_t0.250_c640.000','howe_t0.400_c160.000',oracle['method'],best_s['method']]
        images=[original,rgb(folder/'input.png'),target]+[rgb(folder/(k+'.png')) for k in keys]
        labels=['原本（損傷前）','損傷入力 '+r['family']+f' {r["strength"]:.2f}','暫定教師：桃は対象外',
            'CSC v2 + ACE','Sauvola w61 / k.20','Howe c640 / thi.25','Howe c160 / thi.4',
            'Howe探索内参考最良（後知恵）','Sauvola探索内参考最良（後知恵）']
        width=1616;height=1900;canvas=Image.new('RGB',(width,height),'#faf9f5');d=ImageDraw.Draw(canvas)
        def text(pos,message,size=18):d.text(pos,message,font=ImageFont.truetype(FONT,size),fill='#203442')
        text((18,12),r['patch_id']+' / '+r['family'],26)
        text((18,51),'開発探索で選んだ難例' if exploratory else '出力を見る前に固定した検証例',20)
        text((18,82),'暫定教師・中央384角の診断。正式な精度ではない。Howeは近似Canny。',17)
        positions=[]
        for i,(image,label) in enumerate(zip(images,labels)):
            x=18+(i%3)*532;y=120+(i//3)*565
            text((x,y),label,17);canvas.paste(Image.fromarray(image),(x,y+35));positions.append([x,y+35,512,512])
        text((18,1838),'全パネル原画素512角・縮小なし。参考最良は暫定教師を見て選んだ出力で、自動選択性能ではない。',16)
        text((18,1866),'損傷で観測困難になった文字の失敗から、学習なら復元可能とは結論づけない。',16)
        path=OUT/'figures'/(identifier+'.png');path.parent.mkdir(exist_ok=True,parents=True);canvas.save(path)
        actual=rgb(path)
        for (x,y,w,h),image in zip(positions,images):assert np.array_equal(actual[y:y+h,x:x+w],image)
        figures.append(dict(id=identifier,path=rel(path),sha256=sha(path),case=r['folder'],exploratory=exploratory,
            howe_reference_best=oracle['method'],sauvola_reference_best=best_s['method'],panels=positions))
    dump(OUT/'figure_verification.json',figures)
    dump(OUT/'verification.json',dict(inputs=78,method_outputs=1092,figures=len(figures),all_hashes_checked=True,
        group_splits_disjoint=True,source_and_targets_unchanged=True,deterministic_damage=True,
        actual_known_ink_changed_all_nonidentity=True,matched_shapes_across_severity=True,
        final_test=False,formal_accuracy=False))
    print('Verified 78 inputs,1092 method outputs,8 native comparison PNGs',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','finish']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args.workers)
    else:finish()
