"""Fixed donor quilt: compare fine-L energy correction and separate feather."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_observed_absolute_paper_patches as prior
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/paper_texture_blend_energy_pilot_20261001'
MODES=['energy8','texture4_energy']
SETTINGS=dict(pages=29,ROIs=90,fixed='current mask, direction holds, applied, median-qualified paper, absolute24px donor quilt and source maps',
    control='previous observed_Lab field and alpha recomposed exactly before changes',
    decomposition='per connected selected component, reference-normalized finite9x9 Gaussian sigma1.2 of baseline/quilt L',
    tone='same original alpha8 absolute Lab blend, including a/b',
    energy='target variance=(1-beta)*var_baseline+beta*var_quilt; actual variance from weights and measured component covariance',
    gain='sqrt(target/actual), clipped[1,sqrt2]; zero actual <=1e-12 uses1; statistical model, not a guaranteed output variance',
    beta=dict(energy8='same alpha8',texture4_energy='clip((distance-1)/4,0,1), zero without qualified24px donor'),
    update='same first inside pixel fixed and alpha>0 geometry as absolute trial',
    no_new_donors=True,no_new_noise=True,new_DDRM=False,pixel_GT=False,development=True)


def low(L,component):
    weight=cv2.GaussianBlur(component.astype(np.float32),(9,9),1.2)
    value=cv2.GaussianBlur(np.where(component,L,0),(9,9),1.2)/np.maximum(weight,1e-6)
    return value


def infer(baseline,q,app,sx,alpha):
    selected=app&(sx>=0);control=baseline+(q-baseline)*alpha[:,:,None]
    fields={m:control.copy() for m in MODES};gains={m:np.ones(app.shape,np.float32) for m in MODES};betas={m:np.zeros(app.shape,np.float32) for m in MODES}
    lowbase=np.zeros(app.shape,np.float32);lowq=lowbase.copy();hpbase=lowbase.copy();hpq=lowbase.copy();notes=[]
    distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)
    n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8),connectivity=4)
    for k in range(1,n):
        component=labels==k;bl=low(baseline[:,:,0],component);ql=low(q[:,:,0],component)
        bh=baseline[:,:,0]-bl;qh=q[:,:,0]-ql
        lowbase[component]=bl[component];lowq[component]=ql[component];hpbase[component]=bh[component];hpq[component]=qh[component]
        b=bh[component].astype(np.float64);d=qh[component].astype(np.float64)
        varb=float(np.var(b));varq=float(np.var(d));cov=float(np.mean((b-b.mean())*(d-d.mean())))
        record=dict(component=k,xywh=stats[k,:4].astype(int).tolist(),area=int(stats[k,4]),variance_baseline=varb,variance_quilt=varq,covariance=cov,modes={})
        for mode in MODES:
            beta=alpha if mode=='energy8' else np.clip((distance-1)/4.,0,1)*selected
            target=(1-beta)*varb+beta*varq
            actual=(1-beta)**2*varb+beta**2*varq+2*beta*(1-beta)*cov
            gain=np.ones_like(alpha);valid=actual>1e-12
            gain[valid]=np.sqrt(np.maximum(target[valid],0)/actual[valid]);gain=np.clip(gain,1,np.sqrt(2))
            mixed=(1-beta)*bh+beta*qh;ordinary=(1-alpha)*bh+alpha*qh
            fields[mode][:,:,0][component]+=((mixed*gain)-ordinary)[component]
            gains[mode][component]=gain[component];betas[mode][component]=beta[component]
            record['modes'][mode]=dict(gain_min=float(gain[component].min()),gain_max=float(gain[component].max()),gain_mean=float(gain[component].mean()),
                gain_capped_pixels=int((gain[component]>=np.sqrt(2)-1e-6).sum()))
        notes.append(record)
    # No update occurs where original alpha is zero, regardless of decomposition rounding.
    for mode in MODES:fields[mode][alpha==0]=baseline[alpha==0]
    return fields,gains,betas,dict(baseline_low_L=lowbase,quilt_low_L=lowq,baseline_high_L=hpbase,quilt_high_L=hpq),notes,control


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),str(Path(prior.__file__).relative_to(ROOT)):sha(Path(prior.__file__))}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=json.loads((prior.OUT/'diagnostics.json').read_text());results=[];poison=[]
    for row in rows:
        rid=row['id'];source=prior.OUT/'native'/rid;dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True)
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        original=read(source/'original.png');base=read(source/'baseline.png');old=read(source/'observed_Lab.png')
        app=read(source/'applied_context.png',True);known=read(source/'reference_context.png',True);update=read(source/'update_context.png',True);notes=[]
        outputs={m:base.copy() for m in MODES};stats={}
        if app[s].any():
            baseline=np.load(source/'baseline_field_Lab.npy',allow_pickle=False);q=np.load(source/'quilt_Lab.npy',allow_pickle=False)
            sx=np.load(source/'source_x.npy',allow_pickle=False);alpha=np.load(source/'alpha.npy',allow_pickle=False)
            fields,gains,betas,signals,notes,control=infer(baseline,q,app,sx,alpha)
            assert np.array_equal(control,np.load(source/'observed_Lab_field_Lab.npy',allow_pickle=False))
            assert np.array_equal(prior.fill.frozen.compose(control,update,base,s),old)
            poisoned_base=baseline.copy();poisoned_base[~app]=(250,-220,150);poisoned_q=q.copy();poisoned_q[sx<0]=(250,-220,150)
            repeated=infer(poisoned_base,poisoned_q,app,sx,alpha);assert notes==repeated[4]
            for mode in MODES:
                assert np.array_equal(fields[mode][app],repeated[0][mode][app])
                assert np.array_equal(gains[mode],repeated[1][mode]) and np.array_equal(betas[mode],repeated[2][mode])
                outputs[mode]=prior.fill.frozen.compose(fields[mode],update,base,s)
                assert np.array_equal(fields[mode][:,:,1:],control[:,:,1:])
                assert np.array_equal(outputs[mode][~update[s]],base[~update[s]])
                for name,a in dict(field_Lab=fields[mode],gain=gains[mode],beta=betas[mode]).items():np.save(dest/(mode+'_'+name+'.npy'),a,allow_pickle=False)
            for name,a in signals.items():np.save(dest/(name+'.npy'),a,allow_pickle=False)
            poison.append(rid)
        else:assert np.array_equal(old,base)
        for name,a in dict(original=original,baseline=base,absolute_Lab=old,applied_context=app,reference_context=known,update_context=update,**outputs).items():write(dest/(name+'.png'),a)
        for mode,image in dict(baseline=base,absolute_Lab=old,**outputs).items():stats[mode]=prior.metrics(image,base,app,known,s)
        for mode in MODES:stats[mode]['changed_vs_absolute_Lab']=int(np.any(outputs[mode]!=old,axis=2).sum())
        dump(dest/'components.json',notes)
        figure([original,base,old]+[outputs[m] for m in MODES],['原画像','現行補修','前回・絶対Lab混合','同α8・細模様量補正','紙色8px・細模様4px'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 観測模様の混合減衰',
            '同じ元パッチ配置。元と供給の分散・共分散から補正、倍率は√2まで。境界最初とa/bは前回固定。',
            '新しいノイズ・パッチ・マスクは追加なし。統計補正は自然さや元模様の正解保証ではない。')
        results.append({**row,'blend_energy_diagnostics':stats,'blend_energy_components':notes})
        dump(OUT/'progress.json',dict(completed=len(results),total=90,latest=rid));print('Fine paper blend:',len(results),'/90',rid,[(m,stats[m]['changed_vs_absolute_Lab']) for m in MODES],flush=True)
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',ROIs=90,pages=29,all_absolute_Lab_control_outputs_reproduced=True,
        outside_component_baseline_and_unselected_quilt_poison_invariance=poison,source_quilt_fixed=True,ab_exact_absolute_Lab_float=True,
        first_inside_and_outside_update_exact_baseline=True,new_DDRM=False,pixel_GT=False))
    for p,h in deps.items():assert sha(ROOT/p)==h
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: fine paper blending90 ROIs',flush=True)


if __name__=='__main__':main()
