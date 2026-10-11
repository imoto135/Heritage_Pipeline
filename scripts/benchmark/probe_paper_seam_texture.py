"""Frozen harmonic fields: separate boundary tone from observed-paper texture.

Compare weighted overlap and minimum-error seams with exactly the same donor
patch sequence. Transfer only known-paper L residuals, without DDRM/noise.
"""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_known_paper_harmonic_fill_v2 as prior
from build_npj_v6_comparison import dump, read, sha, write
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/paper_seam_texture_pilot_20261001'
MODES=['boundary','texture_average','texture_seam','combined']
SETTINGS=dict(mask_fixed=True,applied_fixed=True,reference_fixed=True,
    baseline='saved harmonic boundary_smooth image and float Lab field',
    boundary='blend toward saved boundary_raw field inside first 8px; no new solve',
    donor=dict(patch_px=24,stride_px=8,max_candidates=512,min_candidates=4,
        gate='entire patch inside saved known-paper reference',
        signal='L minus reference-only Gaussian sigma6; subtract each patch mean; clip to +/-3 L'),
    quilting=dict(overlap_px=8,tile_stride_px=16,
        first_patch='median patch L residual standard deviation',
        cost='overlap mean squared L error plus .02 times reuse count',
        seam='row-wise minimum-error vertical and horizontal cuts; joint keep-old union',
        comparator='identical selected donor sequence/placements with weighted overlap average'),
    texture_feather='zero at first inside pixel, full at distance>=5px',
    chroma_texture=False,random_noise=False,DDRM_residual=False,
    original_damage_blending=False,pixel_GT=False,training=False,inference=False,
    evaluation='all previous 22 development ROIs; no semantic ROI-label gating')


def minimum_path(error):
    h,w=error.shape;cost=error.astype(np.float64).copy();parent=np.zeros((h,w),np.int32)
    for y in range(1,h):
        for x in range(w):
            start,end=max(0,x-1),min(w,x+2)
            p=start+int(np.argmin(cost[y-1,start:end]))
            parent[y,x]=p;cost[y,x]+=cost[y-1,p]
    path=np.empty(h,np.int32);path[-1]=int(np.argmin(cost[-1]))
    for y in range(h-1,0,-1):path[y-1]=parent[y,path[y]]
    return path


def donors(lab,reference):
    h,w=reference.shape;integral=cv2.integral(reference.astype(np.uint8));positions=[]
    for y in range(0,h-23,8):
        for x in range(0,w-23,8):
            area=integral[y+24,x+24]-integral[y,x+24]-integral[y+24,x]+integral[y,x]
            if area==576:positions.append((x,y))
    if len(positions)>512:positions=[positions[i] for i in np.linspace(0,len(positions)-1,512,dtype=int)]
    if len(positions)<4:return None,positions
    light=lab[:,:,0];weight=cv2.GaussianBlur(reference.astype(np.float32),(0,0),6)
    low=cv2.GaussianBlur(np.where(reference,light,0),(0,0),6)/np.maximum(weight,1e-6)
    residual=light-low;patches=[]
    for x,y in positions:
        patch=residual[y:y+24,x:x+24].copy();patch-=float(patch.mean())
        patches.append(np.clip(patch,-3,3))
    return np.stack(patches),positions


def quilt(patches,positions,shape):
    h,w=shape;H,W=max(h,24)+24,max(w,24)+24
    image=np.zeros((H,W),np.float32);occupied=np.zeros((H,W),bool)
    total=np.zeros((H,W),np.float32);weights=np.zeros((H,W),np.float32)
    source_x=np.full((H,W),-1,np.int32);source_y=np.full((H,W),-1,np.int32)
    use=np.zeros(len(patches),np.int32);records=[]
    axis=np.minimum(1,np.minimum(np.arange(24)+1,24-np.arange(24))/8.)
    tile_weight=(axis[:,None]*axis[None,:]).astype(np.float32)
    first=int(np.argsort(np.std(patches,axis=(1,2)))[len(patches)//2])
    yy,xx=np.indices((24,24))
    for y in range(0,max(h,24),16):
        for x in range(0,max(w,24),16):
            old=image[y:y+24,x:x+24];overlap=occupied[y:y+24,x:x+24]
            if overlap.any():
                cost=np.mean((patches[:,overlap]-old[overlap])**2,axis=1)+.02*use
                chosen=int(np.argmin(cost))
            else:chosen=first
            patch=patches[chosen];error=(old-patch)**2;take=np.ones((24,24),bool)
            if x>0:
                path=minimum_path(error[:,:8]);take[:,:8]&=np.arange(8)[None,:]>path[:,None]
            if y>0:
                path=minimum_path(error[:8,:].T);take[:8,:]&=np.arange(8)[:,None]>path[None,:]
            take|=~overlap
            old[take]=patch[take];occupied[y:y+24,x:x+24]=True
            px,py=positions[chosen]
            sx=source_x[y:y+24,x:x+24];sy=source_y[y:y+24,x:x+24]
            sx[take]=(xx+px)[take];sy[take]=(yy+py)[take]
            total[y:y+24,x:x+24]+=patch*tile_weight;weights[y:y+24,x:x+24]+=tile_weight
            use[chosen]+=1;records.append(dict(x=x,y=y,donor=chosen,source_xy=[px,py]))
    average=total[:h,:w]/np.maximum(weights[:h,:w],1e-6)
    assert (source_x[:h,:w]>=0).all()
    return image[:h,:w],average,source_x[:h,:w],source_y[:h,:w],records


def infer(lab,reference,applied,raw_field,smooth_field):
    distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)
    boundary_weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied
    texture_weight=np.clip((distance-1)/4.,0,1)*applied
    boundary=smooth_field+(raw_field-smooth_field)*boundary_weight[:,:,None]
    patches,positions=donors(lab,reference)
    seam=np.zeros(applied.shape,np.float32);average=seam.copy()
    source_x=np.full(applied.shape,-1,np.int32);source_y=source_x.copy();notes=[]
    if patches is not None:
        n,labels,stats,_=cv2.connectedComponentsWithStats(applied.astype(np.uint8),connectivity=4)
        for i in range(1,n):
            x,y,w,h,area=map(int,stats[i]);component=labels==i
            q,a,sx,sy,records=quilt(patches,positions,(h,w));local=component[y:y+h,x:x+w]
            seam[y:y+h,x:x+w][local]=q[local];average[y:y+h,x:x+w][local]=a[local]
            source_x[y:y+h,x:x+w][local]=sx[local];source_y[y:y+h,x:x+w][local]=sy[local]
            notes.append(dict(id=i,xywh=[x,y,w,h],area=area,tiles=records,
                distinct_donors=len({r['donor'] for r in records})))
        selected=source_x>=0
        assert reference[source_y[selected],source_x[selected]].all()
    fields={m:smooth_field.copy() for m in MODES}
    fields['boundary']=boundary
    fields['texture_average'][:,:,0]+=average*texture_weight
    fields['texture_seam'][:,:,0]+=seam*texture_weight
    fields['combined']=boundary.copy();fields['combined'][:,:,0]+=seam*texture_weight
    return fields,positions,notes,seam,average,source_x,source_y


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        str(Path(prior.__file__).relative_to(ROOT)):sha(Path(prior.__file__)),
        **{str((prior.OUT/name).relative_to(ROOT)):sha(prior.OUT/name) for name in ['manifest.json','complete.json']}}
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)),pages=14,regions=22)
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[];poison_checked=[]
    prior_rows=json.loads((prior.OUT/'diagnostics.json').read_text());originals={}
    for source in prior.previous.mask_probe.old.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:originals[page['page']]=page['original']
    current_page=None;original=None
    for row in prior_rows:
        key=row['id'];page=row['page'];native=prior.OUT/'native'/key
        if current_page!=page:
            original=read(ROOT/originals[page]);current_page=page;gc.collect()
        cx,cy,cw,ch=row['context_xywh'];x,y,w,h=row['xywh'];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        context=original[cy:cy+ch,cx:cx+cw];lab=prior.previous.blending.previous.to_lab(context)
        reference=read(native/'reference_context.png',True);applied=read(native/'applied_context.png',True)
        mask=read(prior.previous.mask_probe.OUT/'pages'/page/'completed/edit.png',True)[cy:cy+ch,cx:cx+cw]
        raw=np.load(native/'boundary_raw_field_Lab.npy',allow_pickle=False)
        smooth=np.load(native/'boundary_smooth_field_Lab.npy',allow_pickle=False)
        base=read(native/'boundary_smooth.png');assert np.array_equal(context[s],read(native/'original.png'))
        if applied[s].any():
            fields,positions,notes,seam,average,sx,sy=infer(lab,reference,applied,raw,smooth)
            poisoned=context.copy();poisoned[mask]=(213,7,11)
            repeat=infer(prior.previous.blending.previous.to_lab(poisoned),reference,applied,raw,smooth)[0]
            for mode in MODES:assert np.array_equal(fields[mode],repeat[mode])
            poison_checked.append(key)
        else:
            fields={m:smooth.copy() for m in MODES};positions=[];notes=[]
            seam=np.zeros(applied.shape,np.float32);average=seam.copy();sx=np.full(applied.shape,-1,np.int32);sy=sx.copy()
        outputs={}
        for mode,field in fields.items():
            value=base.copy();converted=prior.previous.blending.previous.from_lab(field[s])
            value[applied[s]]=converted[applied[s]];outputs[mode]=value
            assert not np.any(np.any(value!=base,axis=2)&~applied[s])
            assert not np.any(np.any(value!=context[s],axis=2)&~mask[s])
        dest=OUT/'native'/key
        for name,value in dict(original=context[s],base=base,edit=mask[s],applied=applied[s],
            reference_context=reference,applied_context=applied,**outputs).items():write(dest/(name+'.png'),value)
        for name,value in dict(texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():
            np.save(dest/(name+'.npy'),value,allow_pickle=False)
        dump(dest/'donors.json',dict(context_xywh=row['context_xywh'],patch_xy=positions,components=notes))
        diagnostics={};distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5)[s]
        core=applied[s]&(distance>=6);known=reference[s];target=applied[s]
        for mode,image in dict(base=base,**outputs).items():
            result_lab=prior.previous.blending.previous.to_lab(image);L=result_lab[:,:,0]
            hp=L-cv2.GaussianBlur(L,(0,0),1.2);jumps=[]
            for a,b,ra,rb,delta in [(target[:-1],target[1:],known[:-1],known[1:],result_lab[:-1]-result_lab[1:]),
                (target[:,:-1],target[:,1:],known[:,:-1],known[:,1:],result_lab[:,:-1]-result_lab[:,1:])]:
                pairs=(a&rb)|(b&ra)
                if pairs.any():jumps.append(np.linalg.norm(delta[pairs],axis=1))
            values=np.concatenate(jumps) if jumps else np.empty(0)
            diagnostics[mode]=dict(boundary_pairs=len(values),median_boundary_deltaLab=float(np.median(values)) if len(values) else None,
                core_L_highpass_std=float(np.std(hp[core])) if core.any() else None,
                L_le_80_applied_pixels=int(((L<=80)&target).sum()),changed_vs_base=int(np.any(image!=base,axis=2).sum()))
        prior.previous.prior_ddrm.figure([context[s],base,outputs['boundary'],outputs['texture_seam'],outputs['combined']],
            ['原画像','前回の紙色補間','境界紙色のみ','紙模様のみ・継ぎ目','境界＋紙模様'],
            OUT/'comparisons'/(key+'.png'),row['title']+' · 境界と紙模様を分離',
            f'マスクと補間場固定。参照24px角パッチ {len(positions)}個。模様は明度だけ。',
            '再推論・再学習・原画像の損傷の混合なし。紙面の観測模様を使う。')
        prior.previous.prior_ddrm.figure([context[s],base,outputs['texture_average'],outputs['texture_seam']],
            ['原画像','前回の紙色補間','同じパッチ・重複平均','同じパッチ・継ぎ目'],
            OUT/'overlap_ablation'/(key+'.png'),row['title']+' · 同一パッチ配置の比較',
            'パッチ選択と配置は完全に同じ。重複平均と最小誤差の継ぎ目だけを変える。',
            '前回の異なるマスク・DDRM上の模様試作とは直接の要素比較にしない。')
        rows.append(dict(id=key,page=page,title=row['title'],xywh=row['xywh'],context_xywh=row['context_xywh'],
            applied_pixels=int(target.sum()),donor_patches=len(positions),components=notes,diagnostics=diagnostics))
        print('Paper seams:',len(rows),'/22',key,'donors',len(positions),flush=True)
    dump(OUT/'diagnostics.json',rows)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',regions=22,pages=14,mask_and_harmonic_fields_fixed=True,
        original_damage_poison_invariance=poison_checked,source_pixels_known_paper=True,
        matched_patch_sequence=True,outside_applied_changed_pixels=0,outside_edit_changed_pixels=0,
        random_noise=False,DDRM_recomputed=False,pixel_GT=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.npy'] and p.name not in ['manifest.json','complete.json']}))
    print('Complete: boundary and matched seam texture, all 22 ROIs',flush=True)


if __name__=='__main__':main()
