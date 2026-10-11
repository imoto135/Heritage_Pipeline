"""Independently replay donor choice, seam provenance, and patch compositing."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import probe_observed_absolute_paper_patches as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/observed_absolute_paper_patches_review_20261001'


def cut(error):
    h,w=error.shape;cost=error.astype(np.float64).copy();parent=np.zeros((h,w),np.int32)
    for y in range(1,h):
        candidates=np.stack([np.r_[np.inf,cost[y-1,:-1]],cost[y-1],np.r_[cost[y-1,1:],np.inf]])
        choice=np.argmin(candidates,axis=0);parent[y]=np.arange(w)+choice-1
        cost[y]+=np.min(candidates,axis=0)
    path=np.zeros(h,np.int32);path[-1]=np.argmin(cost[-1])
    for y in range(h-1,0,-1):path[y-1]=parent[y,path[y]]
    return path


def expected_positions(known):
    h,w=known.shape;positions=[]
    # Direct patch sums rather than the runner's integral image.
    for y in range(0,h-23,8):
        for x in range(0,w-23,8):
            if known[y:y+24,x:x+24].all():positions.append([x,y])
    if len(positions)>512:positions=[positions[i] for i in np.linspace(0,len(positions)-1,512,dtype=int)]
    return positions


def verify_quilt(lab,known,app,positions,records,q,sx,sy):
    assert positions==expected_positions(known)
    if len(positions)<4:
        assert not records and not q.any() and (sx==-1).all() and (sy==-1).all();return []
    patches=np.stack([lab[y:y+24,x:x+24] for x,y in positions])
    n,labels,stats,_=cv2.connectedComponentsWithStats(app.astype(np.uint8),connectivity=4)
    assert len(records)==n-1;expected_q=np.zeros_like(q);expected_sx=np.full(app.shape,-1,np.int32);expected_sy=expected_sx.copy();checked=[]
    yy,xx=np.indices((24,24))
    for k,record in enumerate(records,1):
        x,y,w,h,area=map(int,stats[k]);component=labels==k
        assert record['id']==k and record['xywh']==[x,y,w,h] and record['area']==area
        distance=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5);ring=known&(distance<=64)
        target=np.median(lab[ring],axis=0);assert record['ring_pixels']==int(ring.sum()) and record['target_Lab']==target.tolist()
        image=np.zeros((max(h,24)+24,max(w,24)+24,3),np.float32);occupied=np.zeros(image.shape[:2],bool)
        sourcex=np.full(occupied.shape,-1,np.int32);sourcey=sourcex.copy();uses=np.zeros(len(patches),np.int32)
        first=np.argmin(np.mean((np.mean(patches,axis=(1,2))-target)**2,axis=1));index=0
        for ty in range(0,max(h,24),16):
            for tx in range(0,max(w,24),16):
                tile=record['tiles'][index];index+=1
                assert tile['x']==tx and tile['y']==ty
                old=image[ty:ty+24,tx:tx+24];overlap=occupied[ty:ty+24,tx:tx+24]
                donor=int(np.argmin(np.mean((patches[:,overlap]-old[overlap])**2,axis=(1,2))+.02*uses)) if overlap.any() else int(first)
                assert tile['donor']==donor and tile['source_xy']==positions[donor]
                patch=patches[donor];error=np.mean((old-patch)**2,axis=2);take=np.ones((24,24),bool)
                if tx>0:take[:,:8]&=np.arange(8)[None,:]>cut(error[:,:8])[:,None]
                if ty>0:take[:8,:]&=np.arange(8)[:,None]>cut(error[:8,:].T)[None,:]
                take|=~overlap;old[take]=patch[take];occupied[ty:ty+24,tx:tx+24]=True
                px,py=positions[donor];sourcex[ty:ty+24,tx:tx+24][take]=(px+xx)[take];sourcey[ty:ty+24,tx:tx+24][take]=(py+yy)[take];uses[donor]+=1
        assert index==len(record['tiles'])
        local=component[y:y+h,x:x+w]
        expected_q[y:y+h,x:x+w][local]=image[:h,:w][local]
        expected_sx[y:y+h,x:x+w][local]=sourcex[:h,:w][local];expected_sy[y:y+h,x:x+w][local]=sourcey[:h,:w][local]
        checked.append(dict(component=k,area=area,tiles=index,distinct_donors=int((uses>0).sum())))
    assert np.array_equal(expected_q,q) and np.array_equal(expected_sx,sx) and np.array_equal(expected_sy,sy)
    selected=sx>=0;assert not np.any(selected&~app) and known[sy[selected],sx[selected]].all()
    assert np.array_equal(q[selected],lab[sy[selected],sx[selected]])
    return checked


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());checked=[];unchanged={m:[] for m in prior.MODES};current=None
    for row in rows:
        key=row['page'];rid=row['id'];native=prior.OUT/'native'/rid;source=ROOT/row['local']
        if current!=key:
            original=read(ROOT/row['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
            raw_ok=read(ROOT/'tmp_work/raw_qualified_paper_colour_pilot_20261001/pages'/key/'raw_Lab_ok.png',True);current=key
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=full_lab[cs];base=read(native/'baseline.png');app=read(native/'applied_context.png',True);known=read(native/'reference_context.png',True);update=read(native/'update_context.png',True)
        assert np.array_equal(base,read(source/row['output_file'])) and np.array_equal(read(native/'original.png'),original[y:y+h,x:x+w])
        assert np.array_equal(app,read(source/row['applied_file'],True)) and np.array_equal(known,read(source/'reference_context.png',True)) and not np.any(app&known)
        record=json.loads((native/'donors.json').read_text());assert len(record['patch_xy'])==row['donor_patches']
        provenance=[];source_fail=0;source_dark=0;unique=0
        if app[s].any():
            baseline=np.load(native/'baseline_field_Lab.npy',allow_pickle=False);q=np.load(native/'quilt_Lab.npy',allow_pickle=False)
            sx=np.load(native/'source_x.npy',allow_pickle=False);sy=np.load(native/'source_y.npy',allow_pickle=False)
            assert np.array_equal(baseline,np.load(ROOT/row['field_source'],allow_pickle=False))
            provenance=verify_quilt(lab,known,app,record['patch_xy'],record['components'],q,sx,sy)
            distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)
            alpha=(1-np.exp(-(np.maximum(distance-1,0)/8.)**2))*app*(sx>=0)
            assert np.array_equal(alpha,np.load(native/'alpha.npy',allow_pickle=False)) and np.array_equal(update,alpha>0)
            selected=(sx>=0)&update;selected_roi=selected.copy();roi_mask=np.zeros_like(app);roi_mask[s]=True;selected_roi&=roi_mask
            source_fail=int((~raw_ok[cs][sy[selected_roi],sx[selected_roi]]).sum())
            source_dark=int((lab[sy[selected_roi],sx[selected_roi],0]<50).sum())
            unique=len(np.unique(np.stack([sy[selected_roi],sx[selected_roi]],axis=1),axis=0))
        else:assert not update.any() and not record['patch_xy'] and not record['components']
        for mode in prior.MODES:
            value=read(native/(mode+'.png'))
            if app[s].any():
                field=baseline.copy()
                if mode=='observed_L':field[:,:,0]+=(q[:,:,0]-baseline[:,:,0])*alpha
                else:field+=(q-baseline)*alpha[:,:,None]
                assert np.array_equal(field,np.load(native/(mode+'_field_Lab.npy'),allow_pickle=False))
                if mode=='observed_L':assert np.array_equal(field[:,:,1:],baseline[:,:,1:])
                assert np.array_equal(field[alpha==0],baseline[alpha==0])
                converted=cv2.cvtColor(field[s].astype(np.float32),cv2.COLOR_LAB2BGR);converted=np.rint(np.clip(converted,0,1)*255).astype(np.uint8)
                expected=base.copy();expected[update[s]]=converted[update[s]];assert np.array_equal(expected,value)
            else:assert np.array_equal(value,base)
            assert np.array_equal(value[~update[s]],base[~update[s]])
            changed=int(np.any(value!=base,axis=2).sum());assert changed==row['patch_diagnostics'][mode]['changed_vs_base']
            if changed==0:unchanged[mode].append(rid)
        assert row['applied_ROI']==int(app[s].sum()) and row['update_ROI']==int(update[s].sum())
        checked.append(dict(id=rid,applied_ROI=int(app[s].sum()),update_ROI=int(update[s].sum()),donors=row['donor_patches'],
            provenance=provenance,unique_source_pixels_ROI=unique,source_transfers_failing_raw_Lab_qualification=source_fail,source_transfers_raw_L_lt50=source_dark))
        print('Audit absolute paper:',len(checked),'/90',rid,'raw qualification outlier transfers',source_fail,flush=True)
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['hole_edge_new','new_margin_hole','unseen_thin_crack','extra_top_branch','extra_second_branch','extra_brown_stain','new_brown_stamp','unseen_fold','200021712_00082_1_external_rim_core_10']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 絶対観測紙パッチ',
            ['同じマスク・保留・供給パッチ配置。Lのみ／Lab全色を比較。','中央値資格は生画素の完全無損傷を保証しない。変更画素数を正解率としない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,conditions=180,records=checked,unchanged_ROIs=unchanged,
        entire_patch_qualification_and_full_seam_source_replay=True,donor_cost_and_ring_target_independently_checked=True,
        raw_source_values_exact=True,float_L_only_ab_exact_baseline=True,first_inside_and_outside_update_exact_baseline=True,
        matched_source_maps=True,new_DDRM=False,pixel_GT=False))
    notes=ROOT/'tmp_work/observed_absolute_paper_patches_review_notes_20261001.md'
    if not notes.exists():
        print('Audit ready; create visual review notes before delivery',flush=True);return
    (OUT/'report.md').write_text(notes.read_text());pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==99
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==99
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    raw=ROOT/'tmp_work/raw_qualified_paper_colour_pilot_20261001'
    for n in ['manifest.json','complete.json']:deps[str((raw/n).relative_to(ROOT))]=sha(raw/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=99,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered absolute observed paper patches:99 PNGs /180 conditions',flush=True)


if __name__=='__main__':main()
