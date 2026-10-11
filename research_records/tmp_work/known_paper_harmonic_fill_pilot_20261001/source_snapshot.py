"""Paper-only harmonic fill from unedited references, without DDRM residuals.

Keep the completed mask and saved DDRM/Telea outputs fixed. This is a local
restorer diagnostic, not a validated no-writing classifier or new mask.
"""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import check_margin_top_completion_ddrm as previous
import diagnose_margin_top_fill as telea
from build_npj_v6_comparison import dump, read, sha, write
from package_paper_blending_review import check_bundle

ROOT=previous.ROOT
OUT=ROOT/'tmp_work/known_paper_harmonic_fill_pilot_20261001'
MODES=['boundary_raw','boundary_smooth']
SETTINGS=dict(mask='frozen completed edit; no expansion',baseline='saved completed-mask DDRM',
    eligibility='edit intersect saved safe upper domain; exclude long-line guard and warm anchor guard',
    reference='same original-paper Lab gate; outside entire edit and remaining protected pixels',
    context_px=64,component_connectivity=4,min_component_area=64,min_boundary_pairs=16,
    operator='4-neighbor harmonic Lab interpolation, no screening/prior/guidance from DDRM',
    excluded_boundary='zero normal flux at non-reference neighbors and context edges',
    modes=dict(boundary_raw='actual unedited reference Lab',
               boundary_smooth='reference-only normalized Gaussian sigma 6 Lab'),
    DDRM_residual=False,DDRM_feather=False,random_noise=False,texture_transfer=False,
    original_damage_blending=False,full_page_mask_fixed=True,local_output=True,
    ROI_label_gating=False,pixel_GT=False,independent_accuracy=False,training=False,inference=False)


def reference_lowpass(lab,known):
    weight=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
    value=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),6)
    return value/np.maximum(weight[:,:,None],1e-6)


def solve_fields(original_lab,reference,active):
    assert not np.any(reference&active)
    smooth=reference_lowpass(original_lab,reference)
    fields={m:np.zeros_like(original_lab) for m in MODES}
    used=np.zeros(reference.shape,bool);applied=np.zeros(reference.shape,bool);notes=[]
    n,labels,stats,_=cv2.connectedComponentsWithStats(active.astype(np.uint8),connectivity=4)
    H,W=active.shape
    for k in range(1,n):
        x,y,w,h,area=map(int,stats[k])
        if area<64:
            notes.append(dict(id=k,area=area,applied=False,reason='area below 64'));continue
        component=labels==k;yy,xx=np.nonzero(component);N=len(yy)
        index=np.full(active.shape,-1,np.int32);index[yy,xx]=np.arange(N)
        diagonal=np.zeros(N,np.float64);rhs=np.zeros((N,6),np.float64)
        rows=[];cols=[];values=[];boundary_values=[];local_used=np.zeros_like(used)
        for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny,nx=yy+dy,xx+dx;valid=(ny>=0)&(ny<H)&(nx>=0)&(nx<W)
            ii=np.flatnonzero(valid);oy,ox=ny[valid],nx[valid]
            inside=component[oy,ox];ids=ii[inside];diagonal[ids]+=1
            rows.extend(ids.tolist());cols.extend(index[oy[inside],ox[inside]].tolist());values.extend([-1.]*len(ids))
            boundary=(~inside)&reference[oy,ox];ids=ii[boundary]
            diagonal[ids]+=1
            vv=np.concatenate([original_lab[oy[boundary],ox[boundary]],smooth[oy[boundary],ox[boundary]]],axis=1)
            rhs[ids]+=vv;boundary_values.append(vv)
            local_used[oy[boundary],ox[boundary]]=True
        pairs=sum(len(v) for v in boundary_values)
        if pairs<16:
            notes.append(dict(id=k,area=area,applied=False,reason='fewer than 16 paper boundary pairs',boundary_pairs=pairs));continue
        rows.extend(range(N));cols.extend(range(N));values.extend(diagonal.tolist())
        matrix=sparse.coo_matrix((values,(rows,cols)),shape=(N,N)).tocsr()
        solution=spsolve(matrix,rhs)
        assert np.isfinite(solution).all()
        residual=float(np.abs(matrix@solution-rhs).max())
        refs=np.concatenate(boundary_values)
        lower,upper=refs.min(axis=0),refs.max(axis=0)
        assert np.all(solution>=lower-1e-7) and np.all(solution<=upper+1e-7)
        assert residual<1e-6
        for j,mode in enumerate(MODES):fields[mode][yy,xx]=solution[:,j*3:j*3+3].astype(np.float32)
        applied|=component;used|=local_used
        notes.append(dict(id=k,xywh=[x,y,w,h],area=area,applied=True,boundary_pairs=pairs,
            residual_max=residual,reference_min_Lab=lower.tolist(),reference_max_Lab=upper.tolist(),
            fill_min_Lab=solution.min(axis=0).tolist(),fill_max_Lab=solution.max(axis=0).tolist()))
    return fields,applied,used,notes


def compose(fields,applied,baseline,offset):
    x,y=offset;h,w=baseline.shape[:2];s=(slice(y,y+h),slice(x,x+w));mask=applied[s]
    outputs={}
    for mode,field in fields.items():
        result=baseline.copy();converted=previous.blending.previous.from_lab(field[s])
        result[mask]=converted[mask];outputs[mode]=result
    return outputs


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        str(Path(previous.blending.previous.__file__).relative_to(ROOT)):sha(Path(previous.blending.previous.__file__))}
    for folder in [previous.OUT,previous.mask_probe.OUT,telea.OUT]:
        check_bundle(folder)
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)),
        pages=14,regions=22,selection='same all 22 development ROIs; no semantic ROI labels used')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[];poison_checked=[]
    for source in previous.mask_probe.old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            key=page['page'];original=read(ROOT/page['original']);H,W=original.shape[:2]
            directory=previous.mask_probe.OUT/'pages'/key
            edit=read(directory/'completed/edit.png',True)
            domain=read(directory/'features/safe_domain.png',True)
            line_guard=read(directory/'features/long_line_guard.png',True)
            anchor=read(directory/'features/warm_anchor_guard.png',True)
            eligible=edit&domain&~line_guard&~anchor
            protected=np.zeros((H,W),bool)
            for name in ['protect','hold','red','extra']:protected|=read(source/'v6_masks'/key/(name+'.png'),True)
            protected=(protected|anchor)&~edit
            reference=previous.blending.previous.clean_reference(original,edit,protected,domain&~line_guard)[1]
            write(OUT/'reference_pages'/key/'paper.png',reference)
            write(OUT/'reference_pages'/key/'eligible.png',eligible)
            for roi in page['rois']:
                rid=roi['id'];x,y,w,h=roi['xywh'];x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
                context=original[y0:y1,x0:x1];s=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w));crop=context[s]
                local_edit=edit[y0:y1,x0:x1];active=eligible[y0:y1,x0:x1];known=reference[y0:y1,x0:x1]
                base=read(previous.OUT/'native'/rid/'completed.png');offset=(x-x0,y-y0)
                assert np.array_equal(crop,read(previous.OUT/'native'/rid/'original.png'))
                assert not np.any(known&local_edit)
                lab=previous.blending.previous.to_lab(context)
                if active[s].any():
                    fields,applied,used,notes=solve_fields(lab,known,active)
                else:
                    fields={m:np.zeros_like(lab) for m in MODES};applied=np.zeros_like(active);used=np.zeros_like(known);notes=[]
                outputs=compose(fields,applied,base,offset)
                if applied[s].any():
                    poison=context.copy();poison[local_edit]=(11,3,241)
                    f2,a2,_,_=solve_fields(previous.blending.previous.to_lab(poison),known,active)
                    repeat=compose(f2,a2,base,offset)
                    poisoned_base=base.copy();poisoned_base[applied[s]]=(0,0,0)
                    from_poisoned_base=compose(fields,applied,poisoned_base,offset)
                    for mode in MODES:
                        assert np.array_equal(outputs[mode],repeat[mode])
                        assert np.array_equal(outputs[mode],from_poisoned_base[mode])
                    poison_checked.append(rid)
                dest=OUT/'native'/rid
                values=dict(original=crop,ddrm=base,telea=read(telea.OUT/'native'/rid/'completed_telea.png'),
                    edit=edit[y:y+h,x:x+w],eligible=active[s],applied=applied[s],
                    eligible_context=active,applied_context=applied,reference_context=known,used_boundary_context=used,**outputs)
                for name,value in values.items():write(dest/(name+'.png'),value)
                for mode,value in fields.items():np.save(dest/(mode+'_field_Lab.npy'),value,allow_pickle=False)
                dump(dest/'components.json',notes)
                diagnostics={}
                for mode,image in {k:values[k] for k in ['original','ddrm','telea']+MODES}.items():
                    assert not np.any(np.any(image!=crop,axis=2)&~edit[y:y+h,x:x+w]) if mode!='original' else True
                    if mode in MODES:assert not np.any(np.any(image!=base,axis=2)&~applied[s])
                    light=previous.blending.previous.to_lab(image)[:,:,0];selected=applied[s]
                    diagnostics[mode]=dict(pixels=int(selected.sum()),L_le_80_pixels=int(((light<=80)&selected).sum()),
                        median_L=float(np.median(light[selected])) if selected.any() else None,
                        changed_vs_ddrm=int(np.any(image!=base,axis=2).sum()))
                previous.prior_ddrm.figure([values[k] for k in ['original','ddrm','telea']+MODES],
                    ['原画像','固定マスク → DDRM','同じマスク → Telea','紙色補間・直接境界','紙色補間・平滑境界'],
                    OUT/'comparisons'/(rid+'.png'),roi['title']+' · 同じマスクで紙色から復元',
                    f'紙色補間 {int(applied[s].sum()):,}画素。DDRMの低周波・細かい模様とも引き継がない。',
                    'マスク固定。紙色補間とTeleaはDDRMではない。再学習・追加DDRM推論なし。')
                rows.append(dict(id=rid,page=key,title=roi['title'],xywh=roi['xywh'],context_xywh=[x0,y0,x1-x0,y1-y0],
                    eligible_pixels=int(active[s].sum()),applied_pixels=int(applied[s].sum()),components=notes,diagnostics=diagnostics))
                print('Known-paper harmonic:',len(rows),'/22',rid,'applied',int(applied[s].sum()),flush=True)
            del original,edit,domain,line_guard,anchor,eligible,protected,reference;gc.collect()
    dump(OUT/'diagnostics.json',rows)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=14,regions=22,mask_fixed=True,
        reference_excludes_entire_edit=True,original_damage_and_DDRM_prior_poison_invariance=poison_checked,
        harmonic_maximum_principle=True,linear_solver_residual_checked=True,
        outside_edit_changed_pixels=0,DDRM_recomputed=False,texture_transfer=False,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.npy'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete: known-paper harmonic fill, all 22 ROIs',flush=True)


if __name__=='__main__':main()
