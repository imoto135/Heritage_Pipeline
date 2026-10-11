"""Audit component decomposition, covariance correction and exact compositing."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import probe_paper_texture_blend_energy as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/paper_texture_blend_energy_review_20261001'


def smooth(L,mask):
    kernel=cv2.getGaussianKernel(9,1.2,cv2.CV_64F)
    mass=cv2.sepFilter2D(mask.astype(np.float32),cv2.CV_32F,kernel,kernel)
    values=cv2.sepFilter2D(np.where(mask,L,0),cv2.CV_32F,kernel,kernel)
    return values/np.maximum(mass,1e-6)


def check_metrics(image,base,app,known,s,record):
    assert record['changed_vs_base']==int(np.any(image!=base,axis=2).sum())
    lab=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_BGR2LAB);L=lab[:,:,0];core=app[s]&(cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)[s]>=6)
    value=float(np.std((L-cv2.GaussianBlur(L,(0,0),1.2))[core])) if core.any() else None
    assert value==record['core_L_highpass_std'];assert record['applied_L_le_80']==int(((L<=80)&app[s]).sum())
    diffs=[];ref=known[s];target=app[s]
    for ta,tb,ra,rb,d in [(target[:-1],target[1:],ref[:-1],ref[1:],lab[:-1]-lab[1:]),(target[:,:-1],target[:,1:],ref[:,:-1],ref[:,1:],lab[:,:-1]-lab[:,1:])]:
        selected=(ta&rb)|(tb&ra)
        if selected.any():diffs.append(np.linalg.norm(d[selected],axis=1))
    d=np.concatenate(diffs) if diffs else np.empty(0)
    assert record['boundary_pairs']==len(d) and record['median_boundary_deltaLab']==(float(np.median(d)) if len(d) else None)


def synthetic_checks():
    yy,xx=np.indices((32,32));app=np.zeros((32,32),bool);app[5:27,5:27]=True
    field=np.zeros((32,32,3),np.float32);field[:]=(90,3,12);field[:,:,0]+=(np.sin(xx)+np.cos(yy)).astype(np.float32)
    sx=np.where(app,0,-1).astype(np.int32);distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)
    alpha=(1-np.exp(-(np.maximum(distance-1,0)/8.)**2))*app
    result=prior.infer(field,field.copy(),app,sx,alpha)
    for mode in prior.MODES:assert np.allclose(result[0][mode],field,atol=2e-5,rtol=0)
    constant=np.full(field.shape,(90,3,12),np.float32);donor=np.full(field.shape,(92,4,13),np.float32)
    result=prior.infer(constant,donor,app,sx,alpha)
    control=constant+(donor-constant)*alpha[:,:,None]
    for mode in prior.MODES:assert np.allclose(result[0][mode],control,atol=2e-5,rtol=0)
    anti=field.copy();anti[:,:,0]=180-field[:,:,0];result=prior.infer(field,anti,app,sx,alpha)
    for mode in prior.MODES:assert np.isfinite(result[0][mode]).all() and result[1][mode].max()<=np.sqrt(2)+1e-6
    return dict(identical_observed_field_no_amplification=True,constant_tone_no_added_texture=True,opposite_texture_gain_finite_and_capped=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT);synthetic=synthetic_checks()
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());checked=[];unchanged={m:[] for m in prior.MODES};maxerror=0.
    for row in rows:
        rid=row['id'];source=prior.prior.OUT/'native'/rid;native=prior.OUT/'native'/rid;x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        app=read(native/'applied_context.png',True);known=read(native/'reference_context.png',True);update=read(native/'update_context.png',True)
        base=read(native/'baseline.png');old=read(native/'absolute_Lab.png');assert np.array_equal(base,read(source/'baseline.png')) and np.array_equal(old,read(source/'observed_Lab.png'))
        for n,a in [('applied_context',app),('reference_context',known),('update_context',update)]:assert np.array_equal(a,read(source/(n+'.png'),True))
        record=json.loads((native/'components.json').read_text());assert record==row['blend_energy_components']
        if app[s].any():
            baseline=np.load(source/'baseline_field_Lab.npy',allow_pickle=False);q=np.load(source/'quilt_Lab.npy',allow_pickle=False)
            sx=np.load(source/'source_x.npy',allow_pickle=False);alpha=np.load(source/'alpha.npy',allow_pickle=False);control=baseline+(q-baseline)*alpha[:,:,None]
            assert np.array_equal(control,np.load(source/'observed_Lab_field_Lab.npy',allow_pickle=False))
            selected=app&(sx>=0);n,labels,stats,_=cv2.connectedComponentsWithStats(selected.astype(np.uint8),connectivity=4);assert len(record)==n-1
            distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);signals={k:np.zeros(app.shape,np.float32) for k in ['baseline_low_L','quilt_low_L','baseline_high_L','quilt_high_L']}
            reconstructed={m:control.copy() for m in prior.MODES}
            for k,note in enumerate(record,1):
                component=labels==k;bl=smooth(baseline[:,:,0],component);ql=smooth(q[:,:,0],component);bh=baseline[:,:,0]-bl;qh=q[:,:,0]-ql
                for key,a in [('baseline_low_L',bl),('quilt_low_L',ql),('baseline_high_L',bh),('quilt_high_L',qh)]:signals[key][component]=a[component]
                samples=np.stack([bh[component],qh[component]]).astype(np.float64);covariance=np.cov(samples,bias=True)
                varb,varq,cov=float(covariance[0,0]),float(covariance[1,1]),float(covariance[0,1])
                assert np.allclose([varb,varq,cov],[note['variance_baseline'],note['variance_quilt'],note['covariance']],atol=1e-4,rtol=0)
                assert note['component']==k and note['area']==int(component.sum()) and note['xywh']==stats[k,:4].astype(int).tolist()
                for mode in prior.MODES:
                    beta=alpha if mode=='energy8' else np.clip((distance-1)/4.,0,1)*selected
                    target=(1-beta)*varb+beta*varq;actual=(1-beta)**2*varb+beta**2*varq+2*beta*(1-beta)*cov
                    gain=np.ones_like(alpha);valid=actual>1e-12;gain[valid]=np.sqrt(np.maximum(target[valid],0)/actual[valid]);gain=np.clip(gain,1,np.sqrt(2))
                    saved_gain=np.load(native/(mode+'_gain.npy'),allow_pickle=False);saved_beta=np.load(native/(mode+'_beta.npy'),allow_pickle=False)
                    assert np.allclose(gain[component],saved_gain[component],atol=1e-4,rtol=0) and np.array_equal(beta[component],saved_beta[component])
                    m=note['modes'][mode];assert np.allclose([saved_gain[component].min(),saved_gain[component].max(),saved_gain[component].mean()],[m['gain_min'],m['gain_max'],m['gain_mean']],atol=1e-6,rtol=0)
                    assert m['gain_capped_pixels']==int((saved_gain[component]>=np.sqrt(2)-1e-6).sum())
                    reconstructed[mode][:,:,0][component]+=(((1-beta)*bh+beta*qh)*gain-((1-alpha)*bh+alpha*qh))[component]
            for key,a in signals.items():assert np.allclose(a,np.load(native/(key+'.npy'),allow_pickle=False),atol=5e-5,rtol=0)
            for mode in prior.MODES:
                reconstructed[mode][alpha==0]=baseline[alpha==0];field=np.load(native/(mode+'_field_Lab.npy'),allow_pickle=False)
                error=float(np.abs(reconstructed[mode]-field).max());maxerror=max(maxerror,error);assert error<2e-4,error
                assert np.array_equal(field[:,:,1:],control[:,:,1:]) and np.array_equal(field[alpha==0],baseline[alpha==0])
                expected=base.copy();converted=cv2.cvtColor(field[s],cv2.COLOR_LAB2BGR);converted=np.rint(np.clip(converted,0,1)*255).astype(np.uint8);expected[update[s]]=converted[update[s]]
                value=read(native/(mode+'.png'));assert np.array_equal(value,expected)
        else:assert not record and not update.any()
        for mode in ['baseline','absolute_Lab']+prior.MODES:
            value=read(native/(mode+'.png'));check_metrics(value,base,app,known,s,row['blend_energy_diagnostics'][mode])
            assert np.array_equal(value[~update[s]],base[~update[s]])
            if mode in prior.MODES:
                stat=row['blend_energy_diagnostics'][mode];assert stat['changed_vs_absolute_Lab']==int(np.any(value!=old,axis=2).sum())
                if stat['changed_vs_base']==0:unchanged[mode].append(rid)
        checked.append(rid);print('Audit texture blend:',len(checked),'/90',rid,flush=True)
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['unseen_thin_crack','extra_top_branch','extra_second_branch','hole_edge_new','new_margin_hole','extra_brown_stain','new_brown_stamp','unseen_fold']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 模様の平均減衰を分離',
            ['同じ観測パッチと色、L細模様の混ぜ方のみ変更。最大√2。','細模様量の回復を修復成功と数えない。輪郭・淡い色面も表示する。'])
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,conditions=180,records=checked,unchanged_ROIs=unchanged,synthetic=synthetic,
        independent_separable_decomposition_and_covariance_check=True,max_reconstructed_float_field_error=maxerror,
        exact_saved_float_compositing=True,ab_and_first_inside_exact=True,all_absolute_control_outputs_reproduced=True,
        independent_output_metrics=True,source_quilt_fixed=True,new_DDRM=False,pixel_GT=False))
    notes=ROOT/'tmp_work/paper_texture_blend_energy_review_notes_20261001.md'
    if not notes.exists():print('Texture blend audit ready; create notes',flush=True);return
    (OUT/'report.md').write_text(notes.read_text());pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==98
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==98
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=98,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered texture blend energy:98 PNGs /180 conditions',flush=True)


if __name__=='__main__':main()
