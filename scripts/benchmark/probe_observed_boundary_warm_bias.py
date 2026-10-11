"""Small component-color correction from qualified observed references only."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import diagnose_paper_boundary_reference_bias as diagnostic
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=diagnostic.ROOT
OUT=ROOT/'tmp_work/observed_boundary_warm_bias_pilot_20261001'
MODES=['b_only','warm_Lab']
SETTINGS=dict(baseline='frozen directional abstention77 + new4-books13 original-preserving candidates',
    masks_references_applied_texture_fixed=True,guide='known-paper-only normalized Gaussian sigma32',
    bias='boundary residual median minus other known residual median within64px',
    gate='boundary pixels>=16, qualified ring>=256, boundary median b>=1, positive b bias>=1, boundary median a>=0',
    correction_limits=dict(L=[0,3],a=[-1.5,0],b=[-1.5,0]),fade='first inside pixel zero, linear to8px deeper',
    variants=MODES,pages=29,ROIs=90,pixel_GT=False,development=True,new_DDRM=False,
    provenance='guide and residual statistics read known raw pixels only; nonreferences can be poisoned',
    raw_gate='not changed; previous diagnostic median-paper estimate not used for correction')


def corrections(lab,known,app):
    mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),32)
    guide=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),32)/np.maximum(mass[:,:,None],1e-6)
    supported=known&(mass>=.05);residual=np.where(known[:,:,None],lab-guide,0)
    distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);fade=np.clip((distance-1)/8.,0,1)*app
    n,labels,stats,_=cv2.connectedComponentsWithStats(app.astype(np.uint8),connectivity=4)
    result={mode:np.zeros_like(lab) for mode in MODES};notes=[]
    for k in range(1,n):
        component=labels==k
        boundary=(cv2.dilate(component.astype(np.uint8),np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))>0)&supported
        near=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5)<=64
        ring=supported&near&~boundary
        bmedian=np.median(residual[boundary],axis=0) if boundary.any() else None
        rmedian=np.median(residual[ring],axis=0) if ring.any() else None
        bias=bmedian-rmedian if bmedian is not None and rmedian is not None else None
        accepted=bool(boundary.sum()>=16 and ring.sum()>=256 and bmedian[2]>=1 and bias[2]>=1 and bmedian[1]>=0)
        offset=np.clip(-bias,np.array([0,-1.5,-1.5],np.float32),np.array([3,0,0],np.float32)) if accepted else np.zeros(3,np.float32)
        for mode in MODES:
            delta=offset.copy()
            if mode=='b_only':delta[:2]=0
            result[mode][component]=delta[None,:]*fade[component,None]
        notes.append(dict(id=k,xywh=stats[k,:4].tolist(),area=int(stats[k,4]),boundary=int(boundary.sum()),ring=int(ring.sum()),
            boundary_median=bmedian.tolist() if bmedian is not None else None,ring_median=rmedian.tolist() if rmedian is not None else None,
            bias=bias.tolist() if bias is not None else None,accepted=accepted,offset_Lab=offset.tolist()))
    return result,guide,mass,fade,notes


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(diagnostic.OUT)
    rows=json.loads((diagnostic.OUT/'diagnostics.json').read_text())['regions']
    originals={r['page']['page']:r['page']['original'] for r in diagnostic.previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    originals.update({r['page']:r['original'] for r in json.loads((diagnostic.fresh.INPUT/'baseline_review.json').read_text())['cases']})
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for folder in [diagnostic.OUT,diagnostic.previous.texture.OUT]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    manifest=dict(source_sha256=sha(Path(__file__)),dependencies=deps,settings=SETTINGS)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];poison=[];current=None
    for i,row in enumerate(rows,1):
        rid=row['id'];key=row['page']
        if current!=key:original=read(ROOT/originals[key]);current=key;gc.collect()
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=cv2.cvtColor(original[cs].astype(np.float32)/255,cv2.COLOR_BGR2LAB);crop=original[y:y+h,x:x+w]
        source=ROOT/row['local'];app=read(source/row['applied_file'],True);known=read(source/'reference_context.png',True);base=read(source/row['output_file'])
        dest=OUT/'native'/rid;stats={};outputs={};notes=[];field_path=None
        if app[s].any():
            field_path=diagnostic.previous.texture.OUT/'native'/rid/'baseline_field_Lab.npy'
            field=np.load(field_path,allow_pickle=False)
            assert np.array_equal(diagnostic.fresh.fill.frozen.compose(field,app,base,s),base)
            delta,guide,mass,fade,notes=corrections(lab,known,app)
            bad=lab.copy();bad[~known]=(250,-220,150);repeat=corrections(bad,known,app)
            assert np.array_equal(guide,repeat[1]) and np.array_equal(mass,repeat[2]) and notes==repeat[4]
            for mode in MODES:assert np.array_equal(delta[mode],repeat[0][mode])
            poison.append(rid)
            dest.mkdir(parents=True,exist_ok=True);np.save(dest/'baseline_field_Lab.npy',field,allow_pickle=False)
            np.save(dest/'guide_Lab.npy',guide,allow_pickle=False);np.save(dest/'reference_mass.npy',mass,allow_pickle=False);np.save(dest/'fade.npy',fade,allow_pickle=False)
            for mode in MODES:
                md=dest/mode;md.mkdir(parents=True,exist_ok=True);updated=field+delta[mode]
                changed=np.any(delta[mode]!=0,axis=2)&app
                value=diagnostic.fresh.fill.frozen.compose(updated,changed,base,s)
                assert not np.any(np.any(value!=base,axis=2)&~changed[s])
                assert np.array_equal(value[(fade[s]==0)],base[(fade[s]==0)])
                if mode=='b_only':assert np.array_equal(updated[:,:,:2],field[:,:,:2])
                np.save(md/'correction_Lab.npy',delta[mode],allow_pickle=False);np.save(md/'result_field_Lab.npy',updated,allow_pickle=False);write(md/'update_context.png',changed)
                outputs[mode]=value;stats[mode]=dict(changed_vs_base=int(np.any(value!=base,axis=2).sum()),corrected_pixels=int(changed[s].sum()),
                    mean_correction_Lab=delta[mode][app].mean(axis=0).tolist())
        else:
            for mode in MODES:outputs[mode]=base.copy();stats[mode]=dict(changed_vs_base=0,corrected_pixels=0,mean_correction_Lab=[0,0,0])
        for mode,value in outputs.items():write(dest/mode/'output.png',value)
        for name,item in dict(original=crop,baseline=base,reference_context=known,applied_context=app).items():write(dest/(name+'.png'),item)
        dump(dest/'components.json',notes)
        figure([crop,base]+[outputs[m] for m in MODES],['原画像','固定の補修・方向保留','同じ模様・黄色差だけ補正','同じ模様・暖色Lab差を補正'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 資格紙面の境界偏りだけ補正',
            '同じマスク・参照・適用・観測模様。境界の第一内側画素と保留は維持。','補正は原画像の資格ある参照だけから推定。DDRMの再実行なし。')
        results.append({**row,'field_source':str(field_path.relative_to(ROOT)) if field_path else None,'warm_bias_components':notes,'correction_diagnostics':stats})
        print('Known-paper warm bias:',i,'/90',rid,[stats[m]['changed_vs_base'] for m in MODES],flush=True)
    for path,h in deps.items():assert sha(ROOT/path)==h
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',ROIs=90,pages=29,all_nonreference_poison=poison,baseline_recomposition=True,
        outside_update_exact_baseline=True,boundary_first_pixel_exact_baseline=True,known_mask_and_applied_fixed=True,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:90-ROI reference-only boundary warm bias comparison',flush=True)


if __name__=='__main__':main()
