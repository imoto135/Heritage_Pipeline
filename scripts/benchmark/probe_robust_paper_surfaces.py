"""Fixed full-upper mask and saved fine texture; robust observed-paper surfaces."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import restore_full_upper_strong as prior
import probe_paper_color_reference_gate_v2 as legacy
from build_npj_v6_comparison import dump, read, sha, write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/robust_paper_surfaces_pilot_20261001'
MODES={'plane':1,'quadratic':2}
SETTINGS=dict(baseline='frozen pale6 + short_upper, existing16-pair paper fill',
    mask_applied_reference_fixed=True,fine_L_texture_fixed=True,
    correction='robust local paper surface minus saved smooth harmonic color',
    references='same qualified original known paper within64px of each applied component',
    minimum_reference_pixels=256,max_samples=8192,Huber_iterations=5,
    Huber_scale='per-channel1.4826*MAD, minimum .25',condition_limit=10000,
    surface='linear or quadratic normalized xy, bounded by observed1/99 percentiles',
    support='inside convex hull of original qualified reference coordinates only',
    delta_Lab_limits=[3,1.5,1.5],fade='zero first inside pixel, full at distance>=9px',
    observation_validation='16px checkerboard holdout; not damage GT or independent test',
    no_new_DDRM=True,development=True,pixel_GT=False,pages=25,ROIs=73)


def basis(x,y,box,degree):
    x0,y0,x1,y1=box
    u=2*(np.asarray(x,dtype=np.float64)-x0)/max(x1-x0,1)-1
    v=2*(np.asarray(y,dtype=np.float64)-y0)/max(y1-y0,1)-1
    cols=[np.ones_like(u),u,v]
    if degree==2:cols.extend([u*u,u*v,v*v])
    return np.stack(cols,axis=-1)


def sampled(y,x):
    if len(y)>8192:
        idx=np.linspace(0,len(y)-1,8192,dtype=np.int64);return y[idx],x[idx]
    return y,x


def fit(lab,mask,box,degree):
    yy,xx=sampled(*np.nonzero(mask));a=basis(xx,yy,box,degree)
    if len(yy)<256 or np.linalg.matrix_rank(a)<a.shape[1]:return None
    condition=float(np.linalg.cond(a))
    if condition>10000:return None
    values=lab[yy,xx].astype(np.float64);coeff=np.linalg.lstsq(a,values,rcond=None)[0]
    for _ in range(5):
        residual=values-a@coeff
        scale=np.maximum(.25,1.4826*np.median(np.abs(residual-np.median(residual,axis=0)),axis=0))
        weights=np.minimum(1,1.345*scale/np.maximum(np.abs(residual),1e-12))
        for c in range(3):
            root=np.sqrt(weights[:,c]);coeff[:,c]=np.linalg.lstsq(a*root[:,None],values[:,c]*root,rcond=None)[0]
    lo,hi=np.percentile(values,[1,99],axis=0)
    return dict(coefficients=coeff.tolist(),bounds=[lo.tolist(),hi.tolist()],condition=condition,
                samples=len(yy),reference_pixels=int(mask.sum()),box=box,degree=degree)


def predict(model,y,x):
    lo,hi=np.asarray(model['bounds']);coeff=np.asarray(model['coefficients'])
    return np.clip(basis(x,y,model['box'],model['degree'])@coeff,lo,hi).astype(np.float32)


def surfaces(lab,known,applied,degree):
    h,w=known.shape;surface=np.zeros_like(lab);support=np.zeros_like(known);notes=[]
    n,labels,stats,_=cv2.connectedComponentsWithStats(applied.astype(np.uint8),connectivity=4)
    for k in range(1,n):
        x,y,cw,ch,area=map(int,stats[k]);x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(w,x+cw+64),min(h,y+ch+64)
        local=labels[y0:y1,x0:x1]==k
        distance=cv2.distanceTransform((~local).astype(np.uint8),cv2.DIST_L2,5)
        reference=known[y0:y1,x0:x1]&(distance<=64)
        local_lab=lab[y0:y1,x0:x1];box=[0,0,x1-x0-1,y1-y0-1]
        model=fit(local_lab,reference,box,degree)
        note=dict(id=k,xywh=[x,y,cw,ch],area=area,reference_pixels=int(reference.sum()),accepted=model is not None)
        if model is not None:
            ry,rx=np.nonzero(reference);hull=cv2.convexHull(np.stack([rx,ry],axis=1).astype(np.int32))
            inside=np.zeros(reference.shape,np.uint8);cv2.fillConvexPoly(inside,hull,1)
            targets=local&(inside>0);ty,tx=np.nonzero(targets)
            surface[y0:y1,x0:x1][targets]=predict(model,ty,tx);support[y0:y1,x0:x1]|=targets
            yy,xx=np.indices(reference.shape);fold=((xx+x0)//16+(yy+y0)//16)%2
            validation=fit(local_lab,reference&(fold==0),box,degree);held=reference&(fold==1)
            vy,vx=sampled(*np.nonzero(held))
            if validation is not None and len(vy):
                error=predict(validation,vy,vx)-local_lab[vy,vx]
                note['observation_holdout']=dict(train_model=validation,test_samples=len(vy),
                    RMSE_Lab=np.sqrt(np.mean(error.astype(np.float64)**2,axis=0)).tolist(),
                    median_abs_Lab=np.median(np.abs(error),axis=0).tolist())
            note.update(model=model,inside_reference_hull=int(targets.sum()),outside_hull_retained=int((local&~targets).sum()))
        notes.append(note)
    return surface,support,notes


def color_assets(row,lab,known,retained,base,s,previous_rows):
    """Use latest saved fields; never substitute an older field after a failed match."""
    rid=row['id'];latest=prior.OUT/'native'/rid/'short_upper'
    choices=[latest,ROOT/row['baseline_local']]
    if rid in previous_rows:choices.append(ROOT/previous_rows[rid]['baseline_local'])
    for module in [prior.fresh3,prior.fresh4]:choices.append(module.OUT/'native'/rid/'near8')
    anchor=legacy.prior
    choices.extend([anchor.OUT/'native'/rid/'near8',anchor.prior.OUT/'native'/rid,
                    anchor.prior.fill.new.OUT/'native'/rid/'rim'])
    for directory in choices:
        if not (directory/'result_field_Lab.npy').exists():continue
        source_known=read(directory/'reference_context.png',True)
        if not np.array_equal(source_known,known):continue
        applied=read(directory/'applied_context.png',True)
        if np.any(applied&~retained):continue
        field=np.load(directory/'result_field_Lab.npy',allow_pickle=False)
        if not np.array_equal(prior.fill.frozen.compose(field,applied,base,s),base):continue
        fields=dict(np.load(directory/'harmonic_fields_Lab.npz',allow_pickle=False))
        seam=np.load(directory/'texture_seam.npy',allow_pickle=False)
        return directory,fields,field,seam,applied,False
    # The earlier rim run saved combined fields and seams but not raw/smooth fields.
    rim=anchor.prior.fill.frozen.OUT/'native'/rid
    harmonic=anchor.prior.fill.frozen.prior.prior.OUT/'native'/rid
    texture=anchor.prior.fill.frozen.prior.OUT/'native'/rid
    if (rim/'rim_field_Lab.npy').exists():
        assert np.array_equal(known,read(rim/'rim_reference_context.png',True))
        applied=read(rim/'rim_applied_context.png',True)
        fields,actual,_,_=prior.fill.solve_fields(lab,known,applied,16)
        assert np.array_equal(actual,applied)
        field=np.load(rim/'rim_field_Lab.npy',allow_pickle=False);seam=np.load(rim/'texture_seam.npy',allow_pickle=False)
        directory=rim;regenerated=True
    else:
        assert np.array_equal(known,read(harmonic/'reference_context.png',True))
        applied=read(harmonic/'applied_context.png',True)
        fields={name:np.load(harmonic/(name+'_field_Lab.npy'),allow_pickle=False) for name in ['boundary_raw','boundary_smooth']}
        seam=np.load(texture/'texture_seam.npy',allow_pickle=False)
        distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
        weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied
        field=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
        field[:,:,0]+=seam*np.clip((distance-1)/4.,0,1)*applied
        directory=harmonic;regenerated=False
    assert not np.any(applied&~retained)
    assert np.array_equal(prior.fill.frozen.compose(field,applied,base,s),base)
    return directory,fields,field,seam,applied,regenerated


def preflight():
    yy,xx=np.indices((90,110));lab=np.stack([80+.025*xx-.01*yy,2+.002*xx,12+.003*yy],axis=2).astype(np.float32)
    applied=np.zeros((90,110),bool);applied[30:60,35:75]=True;known=~applied
    for degree in MODES.values():
        surface,support,_=surfaces(lab,known,applied,degree)
        assert support[applied].all() and np.allclose(surface[applied],lab[applied],atol=2e-5)
        changed=lab.copy();changed[~known]=(230,-180,160)
        repeat=surfaces(changed,known,applied,degree)
        assert np.array_equal(surface,repeat[0])


def main():
    preflight();OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());assert len(rows)==73
    old={r['id']:r for r in json.loads((prior.previous.OUT/'diagnostics.json').read_text())}
    originals={r['page']['page']:r['page']['original'] for r in prior.mask.pale.prior.descriptors()}
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [prior.OUT,prior.mask.OUT]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for module in [prior,legacy,prior.fill,prior.fill.frozen,prior.fill.frozen.prior,prior.fill.frozen.prior.prior]:
        dependencies[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    for path in originals.values():dependencies[path]=sha(ROOT/path)
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];poison=[];current=None
    for row in rows:
        rid=row['id'];key=row['page']
        if current!=key:original=read(ROOT/originals[key]);current=key;gc.collect()
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        context=original[cy:cy+ch,cx:cx+cw];crop=context[s];lab=cv2.cvtColor(context.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        source=prior.OUT/'native'/rid/'short_upper';dest=OUT/'native'/rid
        base=read(source/'output.png');known=read(source/'reference_context.png',True)
        retained=read(source/'retained_applied_context.png',True);edit=read(source/'edit.png',True)
        outputs={};stats={};asset=None;notes={};applied=np.zeros_like(known)
        if retained[s].any():
            asset,fields,base_field,seam,applied,regenerated=color_assets(row,lab,known,retained,base,s,old)
            distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
            weight=np.clip((distance-1)/8.,0,1)*applied
            reconstruction=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*np.exp(-(np.maximum(distance-1,0)/8.)**2)[:,:,None]*applied[:,:,None]
            reconstruction[:,:,0]+=seam*np.clip((distance-1)/4.,0,1)*applied
            assert np.array_equal(reconstruction,base_field)
            np.savez_compressed((dest.mkdir(parents=True,exist_ok=True) or dest)/'baseline_fields_Lab.npz',**fields)
            for name,item in dict(baseline_field_Lab=base_field,texture_seam=seam).items():np.save(dest/(name+'.npy'),item,allow_pickle=False)
            for path in asset.glob('*'):
                if path.is_file():dependencies[str(path.relative_to(ROOT))]=sha(path)
            for mode,degree in MODES.items():
                surface,support,component_notes=surfaces(lab,known,applied,degree)
                correction=np.clip(surface-fields['boundary_smooth'],-np.array([3,1.5,1.5],np.float32),np.array([3,1.5,1.5],np.float32))*(weight*support)[:,:,None]
                update=applied&support&(weight>0);field=base_field+correction
                value=prior.fill.frozen.compose(field,update,base,s)
                poisoned=lab.copy();poisoned[~known]=(235,-180,140)
                repeat=surfaces(poisoned,known,applied,degree)
                assert np.array_equal(surface,repeat[0]) and np.array_equal(support,repeat[1]) and component_notes==repeat[2]
                assert not np.any(correction[~update]) and not np.any(correction[distance<=1])
                assert np.array_equal(field[:,:,1:][~update],base_field[:,:,1:][~update])
                md=dest/mode;md.mkdir(parents=True,exist_ok=True)
                for name,item in dict(surface_Lab=surface,correction_Lab=correction,result_field_Lab=field).items():np.save(md/(name+'.npy'),item,allow_pickle=False)
                write(md/'support_context.png',support);write(md/'update_context.png',update);dump(md/'components.json',component_notes)
                notes[mode]=component_notes;poison.append(dict(id=rid,mode=mode))
                outputs[mode]=value
                stats[mode]=dict(updated=int(update[s].sum()),changed_vs_base=int(np.any(value!=base,axis=2).sum()),
                    mean_delta_Lab=correction[s][update[s]].mean(axis=0).tolist() if update[s].any() else None,
                    max_abs_delta_Lab=np.max(np.abs(correction),axis=(0,1)).tolist(),model_components=sum(n['accepted'] for n in component_notes))
        else:
            for mode in MODES:outputs[mode]=base.copy();stats[mode]=dict(updated=0,changed_vs_base=0,mean_delta_Lab=None,max_abs_delta_Lab=[0,0,0],model_components=0)
        for mode,value in outputs.items():
            assert not np.any(np.any(value!=base,axis=2)&~applied[s])
            assert not np.any(np.any(value!=crop,axis=2)&~edit)
            write(dest/mode/'output.png',value)
        for name,item in dict(original=crop,baseline=base,reference_context=known,retained_applied_context=retained,color_applied_context=applied,edit=edit).items():write(dest/(name+'.png'),item)
        dump(dest/'color_source.json',dict(path=str(asset.relative_to(ROOT)) if asset else None,
            baseline_exactly_reconstructed=True,retained_outside_source_field=int((retained&~applied)[s].sum()),
            source_fields_regenerated=regenerated if asset else False))
        elsewhere='旧14ページ固定ROIの紙面外は保存DDRM。' if row['base_elsewhere']=='saved_DDRM' else '紙面適用外は原画像。'
        figure([crop,base]+[outputs[m] for m in MODES],['原画像','固定上方回収＋従来紙色','観測紙色・一次面','観測紙色・二次面'],OUT/'comparisons'/(rid+'.png'),
            row['title']+' · マスク・参照・細かいL模様を固定',
            '既知紙面64px内の頑健な低次面。参照凸包内のみ、境界8pxで混合、Lab差を制限。',
            elsewhere+' 新しいDDRMなし。損傷の正解・独立精度評価ではない。')
        results.append({**row,'surface_diagnostics':stats});dump(OUT/'progress.json',dict(completed=len(results),total=73,latest=rid))
        print('Robust paper surfaces:',len(results),'/73',rid,{k:v['changed_vs_base'] for k,v in stats.items()},flush=True)
    dump(OUT/'final_dependencies.json',dependencies)
    for path,value in dependencies.items():assert sha(ROOT/path)==value
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',ROIs=73,variants=146,baseline_exactly_reconstructed=True,
        fine_texture_fixed=True,mask_and_applied_fixed=True,first_inside_pixel_unchanged=True,outside_edit_changes=0,
        all_nonreference_poison_cases=poison,synthetic_linear_surface_recovery=True,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: robust observed paper surfaces,73 ROIs',flush=True)


if __name__=='__main__':main()
