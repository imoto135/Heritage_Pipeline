"""Audit saved surface models and fixed-mask comparisons independently."""
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import probe_robust_paper_surfaces as prior
from build_npj_v6_comparison import dump, read, sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/robust_paper_surfaces_review_v2_20261001'


def evaluate(model,y,x):
    x0,y0,x1,y1=model['box'];u=(np.asarray(x,dtype=np.float64)-x0)*2/max(x1-x0,1)-1
    v=(np.asarray(y,dtype=np.float64)-y0)*2/max(y1-y0,1)-1
    a=np.stack([np.ones_like(u),u,v]+([u*u,u*v,v*v] if model['degree']==2 else []),axis=-1)
    value=a@np.asarray(model['coefficients']);lo,hi=np.asarray(model['bounds'])
    return np.clip(value,lo,hi).astype(np.float32)


def audit_surface(lab,known,applied,native,mode):
    saved=np.load(native/mode/'surface_Lab.npy',allow_pickle=False)
    support=read(native/mode/'support_context.png',True)
    expected=np.zeros_like(saved);expected_support=np.zeros_like(support);checks=[]
    notes=json.loads((native/mode/'components.json').read_text())
    n,labels,stats,_=cv2.connectedComponentsWithStats(applied.astype(np.uint8),connectivity=4)
    assert len(notes)==n-1
    for note in notes:
        k=note['id'];x,y,w,h,area=map(int,stats[k]);assert note['xywh']==[x,y,w,h] and note['area']==area
        x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(lab.shape[1],x+w+64),min(lab.shape[0],y+h+64)
        local=labels[y0:y1,x0:x1]==k
        distance=cv2.distanceTransform((~local).astype(np.uint8),cv2.DIST_L2,5)
        reference=known[y0:y1,x0:x1]&(distance<=64)
        assert int(reference.sum())==note['reference_pixels']
        if not note['accepted']:continue
        model=note['model'];ry,rx=np.nonzero(reference)
        selected=np.linspace(0,len(ry)-1,8192,dtype=np.int64) if len(ry)>8192 else np.arange(len(ry))
        bounds=np.percentile(lab[y0:y1,x0:x1][ry[selected],rx[selected]].astype(np.float64),[1,99],axis=0)
        assert np.array_equal(bounds,np.asarray(model['bounds'])) and model['condition']<=10000
        hull=cv2.convexHull(np.stack([rx,ry],axis=1).astype(np.int32));inside=np.zeros(reference.shape,np.uint8)
        cv2.fillConvexPoly(inside,hull,1);targets=local&(inside>0);ty,tx=np.nonzero(targets)
        expected[y0:y1,x0:x1][targets]=evaluate(model,ty,tx);expected_support[y0:y1,x0:x1]|=targets
        assert int(targets.sum())==note['inside_reference_hull']
        assert int((local&~targets).sum())==note['outside_hull_retained']
        check=dict(component=k,reference_pixels=int(reference.sum()),inside_hull=int(targets.sum()))
        if 'observation_holdout' in note:
            yy,xx=np.indices(reference.shape);held=reference&(((xx+x0)//16+(yy+y0)//16)%2==1)
            vy,vx=np.nonzero(held)
            if len(vy)>8192:
                index=np.linspace(0,len(vy)-1,8192,dtype=np.int64);vy,vx=vy[index],vx[index]
            cv=note['observation_holdout'];error=evaluate(cv['train_model'],vy,vx)-lab[y0:y1,x0:x1][vy,vx]
            rmse=np.sqrt(np.mean(error.astype(np.float64)**2,axis=0));median=np.median(np.abs(error),axis=0)
            assert len(vy)==cv['test_samples']
            assert np.allclose(rmse,cv['RMSE_Lab'],atol=1e-6,rtol=0)
            assert np.allclose(median,cv['median_abs_Lab'],atol=1e-6,rtol=0)
            check.update(known_paper_holdout_RMSE_Lab=rmse.tolist(),test_samples=len(vy))
        checks.append(check)
    assert np.array_equal(expected_support,support)
    assert np.allclose(expected,saved,atol=1e-6,rtol=0)
    return saved,support,checks


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    dependencies=json.loads((prior.OUT/'final_dependencies.json').read_text())
    for path,value in dependencies.items():assert sha(ROOT/path)==value
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());assert len(rows)==73
    originals={r['page']['page']:r['page']['original'] for r in prior.prior.mask.pale.prior.descriptors()}
    current=None;checks=[];unchanged={m:[] for m in prior.MODES}
    for row in rows:
        rid=row['id'];key=row['page']
        if current!=key:original=read(ROOT/originals[key]);current=key
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        crop=original[y:y+h,x:x+w];lab=cv2.cvtColor(original[cy:cy+ch,cx:cx+cw].astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        native=prior.OUT/'native'/rid;source=prior.prior.OUT/'native'/rid/'short_upper'
        base=read(native/'baseline.png');known=read(native/'reference_context.png',True)
        applied=read(native/'color_applied_context.png',True);retained=read(native/'retained_applied_context.png',True)
        edit=read(native/'edit.png',True);assert np.array_equal(base,read(source/'output.png'))
        assert np.array_equal(known,read(source/'reference_context.png',True))
        assert np.array_equal(retained,read(source/'retained_applied_context.png',True))
        assert np.array_equal(edit,read(source/'edit.png',True)) and np.array_equal(crop,read(native/'original.png'))
        assert not np.any(applied&~retained)
        if applied[s].any():
            fields=dict(np.load(native/'baseline_fields_Lab.npz',allow_pickle=False))
            seam=np.load(native/'texture_seam.npy',allow_pickle=False);field=np.load(native/'baseline_field_Lab.npy',allow_pickle=False)
            distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
            bw=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied
            expected=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:,:,None]
            expected[:,:,0]+=seam*np.clip((distance-1)/4.,0,1)*applied
            assert np.array_equal(expected,field) and np.array_equal(prior.prior.fill.frozen.compose(field,applied,base,s),base)
        for mode in prior.MODES:
            value=read(native/mode/'output.png');stat=row['surface_diagnostics'][mode]
            check=dict(id=rid,mode=mode)
            if applied[s].any():
                surface,support,components=audit_surface(lab,known,applied,native,mode)
                weight=np.clip((distance-1)/8.,0,1)*applied
                update=applied&support&(weight>0)
                delta=np.clip(surface-fields['boundary_smooth'],-np.array([3,1.5,1.5],np.float32),np.array([3,1.5,1.5],np.float32))*(weight*support)[:,:,None]
                assert np.array_equal(update,read(native/mode/'update_context.png',True))
                saved_delta=np.load(native/mode/'correction_Lab.npy',allow_pickle=False)
                assert np.array_equal(delta,saved_delta)
                result=np.load(native/mode/'result_field_Lab.npy',allow_pickle=False)
                assert np.array_equal(result,field+delta)
                assert not np.any(delta[distance<=1]) and not np.any(delta[~update])
                assert np.array_equal(prior.prior.fill.frozen.compose(result,update,base,s),value)
                assert not np.any(np.any(value!=base,axis=2)&~update[s])
                assert stat['updated']==int(update[s].sum())
                assert np.allclose(np.max(np.abs(delta),axis=(0,1)),stat['max_abs_delta_Lab'],atol=1e-6,rtol=0)
                check['components']=components
            else:assert np.array_equal(value,base)
            assert stat['changed_vs_base']==int(np.any(value!=base,axis=2).sum())
            assert not np.any(np.any(value!=base,axis=2)&~applied[s])
            assert not np.any(np.any(value!=crop,axis=2)&~edit)
            if stat['changed_vs_base']==0:unchanged[mode].append(rid)
            checks.append(check)
    (OUT/'paper').mkdir(exist_ok=True)
    for path in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(path,OUT/'paper'/path.name)
    focus=['extra_second_branch','extra_top_branch','hole_edge_new','new_margin_hole','unseen_fold','extra_brown_stain','extra_isolated_marks','new_brown_stamp']
    for i,rid in enumerate(focus,1):
        stack([OUT/'paper'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 観測紙色の低次面',
              ['原画像・固定マスクの従来紙色・一次面・二次面。マスクと細かいL模様は同じ。',
               '紙端・印影の反例も保持。色面の観測適合度は損傷領域の正解精度ではない。'])
    uncertain=[r for r in rows if r['page']=='200021712_00082_1' and not r.get('fixed_ROI',False)]
    stack([OUT/'paper'/(r['id']+'.png') for r in uncertain],OUT/'09_unknown_old_components.png',
          '原因未確定の5曲線を保持して同じ紙色比較',
          ['既存マスクの正誤は未確定。今回の紙色だけの変更を損傷回収と数えない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=73,variants=146,checks=checks,
        unchanged_ROIs=unchanged,masks_reference_applied_unchanged=True,
        baseline_field_exactly_reconstructed=True,fine_L_seam_fixed=True,first_inside_pixel_unchanged=True,
        model_and_holdout_reconstruction=True,outside_edit_changes=0,pixel_GT=False,new_DDRM=False))
    notes=ROOT/'tmp_work/robust_paper_surfaces_review_notes_v2_20261001.md'
    (OUT/'report.md').write_text(notes.read_text());pngs=sorted(OUT.rglob('*.png'))
    assert len(pngs)==82
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==len(pngs)
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=len(pngs),
        dependencies={str(notes.relative_to(ROOT)):sha(notes),**{str((prior.OUT/n).relative_to(ROOT)):sha(prior.OUT/n) for n in ['manifest.json','complete.json','final_dependencies.json']}},
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered robust observed paper surfaces:',len(pngs),'PNGs',flush=True)


if __name__=='__main__':main()
