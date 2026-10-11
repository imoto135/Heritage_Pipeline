"""Fixed upper mask and color fields; observed L patches with adaptive scale."""
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import restore_chromatic_core_rims as history
import probe_robust_paper_surfaces as assets
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=history.ROOT
OUT=ROOT/'tmp_work/adaptive_paper_texture_scale_pilot_v2_20261001'
SIZES=[24,16,12]
MODES=['uniform12','adaptive']
texture=history.fill.frozen.prior
SETTINGS=dict(baseline='frozen upper strong + horizontal exclusion; original external4 baselines',
    mask_reference_applied_color_fixed=True,ROIs=77,pages=25,
    patch_sizes=SIZES,minimum_qualified_patches=4,max_patches=512,
    source='every pixel of patch must belong to same qualified original known paper',
    sample_stride='floor(patch_size/3)',overlap='floor(patch_size/3)',tile_stride='patch_size-overlap',
    signal='same original L minus known-only Gaussian sigma6; patch mean removed; clip +/-3',
    quilting='same deterministic median-std first patch, minimum-error seam, .02 reuse penalty',
    uniform12='12px everywhere with4+ qualified patches; otherwise saved texture retained',
    adaptive='largest24/16/12 with4+ qualified patches; none -> saved texture retained',
    boundary='same texture fade zero at first inside pixel, full at distance>=5',
    no_chromatic_rim_addition=True,no_low_order_color_surface=True,
    no_coarse_texture_or_ab_texture=True,no_random_noise=True,new_DDRM=False,development=True,pixel_GT=False)


def donors(lab,known,size):
    h,w=known.shape;step=size//3;integral=cv2.integral(known.astype(np.uint8));positions=[]
    for y in range(0,h-size+1,step):
        for x in range(0,w-size+1,step):
            area=integral[y+size,x+size]-integral[y,x+size]-integral[y+size,x]+integral[y,x]
            if area==size*size:positions.append((x,y))
    total=len(positions)
    if total>512:positions=[positions[i] for i in np.linspace(0,total-1,512,dtype=int)]
    if len(positions)<4:return None,positions,total
    mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),6)
    low=cv2.GaussianBlur(np.where(known,lab[:,:,0],0),(0,0),6)/np.maximum(mass,1e-6)
    residual=lab[:,:,0]-low;patches=[]
    for x,y in positions:
        patch=residual[y:y+size,x:x+size].copy();patch-=float(patch.mean());patches.append(np.clip(patch,-3,3))
    return np.stack(patches),positions,total


def quilt(patches,positions,shape,size):
    h,w=shape;overlap=size//3;stride=size-overlap;H,W=max(h,size)+size,max(w,size)+size
    image=np.zeros((H,W),np.float32);occupied=np.zeros((H,W),bool)
    total=np.zeros((H,W),np.float32);weights=np.zeros((H,W),np.float32)
    sx=np.full((H,W),-1,np.int32);sy=sx.copy();use=np.zeros(len(patches),np.int32);records=[]
    axis=np.minimum(1,np.minimum(np.arange(size)+1,size-np.arange(size))/overlap)
    tile_weight=(axis[:,None]*axis[None,:]).astype(np.float32)
    first=int(np.argsort(np.std(patches,axis=(1,2)))[len(patches)//2]);yy,xx=np.indices((size,size))
    for y in range(0,max(h,size),stride):
        for x in range(0,max(w,size),stride):
            old=image[y:y+size,x:x+size];shared=occupied[y:y+size,x:x+size]
            cost=np.mean((patches[:,shared]-old[shared])**2,axis=1)+.02*use if shared.any() else None
            chosen=int(np.argmin(cost)) if cost is not None else first
            patch=patches[chosen];error=(old-patch)**2;take=np.ones((size,size),bool)
            if x>0:
                path=texture.minimum_path(error[:,:overlap]);take[:,:overlap]&=np.arange(overlap)[None,:]>path[:,None]
            if y>0:
                path=texture.minimum_path(error[:overlap,:].T);take[:overlap,:]&=np.arange(overlap)[:,None]>path[None,:]
            take|=~shared;old[take]=patch[take];occupied[y:y+size,x:x+size]=True
            px,py=positions[chosen];dx=sx[y:y+size,x:x+size];dy=sy[y:y+size,x:x+size]
            dx[take]=(xx+px)[take];dy[take]=(yy+py)[take]
            total[y:y+size,x:x+size]+=patch*tile_weight;weights[y:y+size,x:x+size]+=tile_weight
            use[chosen]+=1;records.append(dict(x=x,y=y,donor=chosen,source_xy=[px,py]))
    return image[:h,:w],total[:h,:w]/np.maximum(weights[:h,:w],1e-6),sx[:h,:w],sy[:h,:w],records


def infer(lab,known,applied,size):
    patches,positions,total=donors(lab,known,size);seam=np.zeros(applied.shape,np.float32);average=seam.copy()
    sx=np.full(applied.shape,-1,np.int32);sy=sx.copy();notes=[]
    if patches is not None:
        n,labels,stats,_=cv2.connectedComponentsWithStats(applied.astype(np.uint8),connectivity=4)
        for k in range(1,n):
            x,y,w,h,area=map(int,stats[k]);local=labels[y:y+h,x:x+w]==k
            q,a,dx,dy,records=quilt(patches,positions,(h,w),size)
            seam[y:y+h,x:x+w][local]=q[local];average[y:y+h,x:x+w][local]=a[local]
            sx[y:y+h,x:x+w][local]=dx[local];sy[y:y+h,x:x+w][local]=dy[local]
            notes.append(dict(id=k,xywh=[x,y,w,h],area=area,tiles=records,distinct_donors=len({r['donor'] for r in records})))
        selected=sx>=0;assert known[sy[selected],sx[selected]].all()
    return seam,average,sx,sy,dict(patch_size=size,patch_xy=positions,total_qualified_patches=total,components=notes)


def preflight():
    # Original24 implementation parity and no qualification change with smaller patches.
    yy,xx=np.indices((80,96));known=np.ones((80,96),bool);applied=np.zeros_like(known);applied[30:42,35:51]=True;known&=~applied
    lab=np.stack([80+np.sin(xx/2.)+.4*np.cos(yy/3.),np.full_like(xx,2),np.full_like(xx,12)],axis=2).astype(np.float32)
    patches,positions,_=donors(lab,known,24);old,op=texture.donors(lab,known)
    assert positions==op and np.array_equal(patches,old)
    assert patches is not None
    for shape in [(12,16),(30,36)]:
        a=quilt(patches,positions,shape,24);b=texture.quilt(old,op,shape)
        for i in range(4):assert np.array_equal(a[i],b[i])
        assert a[4]==b[4]
    for size in SIZES:
        a=infer(lab,known,applied,size);bad=lab.copy();bad[~known]=(235,-180,140);b=infer(bad,known,applied,size)
        for i in range(4):assert np.array_equal(a[i],b[i])
        assert a[4]==b[4]
        for x,y in a[4]['patch_xy']:assert known[y:y+size,x:x+size].all()


def main():
    preflight();OUT.mkdir(parents=True,exist_ok=True)
    for folder in [history.previous.OUT,history.extras.OUT,assets.OUT]:check_bundle(folder)
    rows=history.regions();assert len(rows)==77
    asset_rows={r['id']:r for r in json.loads((history.previous.OUT/'diagnostics.json').read_text())}
    old={r['id']:r for r in json.loads((history.previous.previous.OUT/'diagnostics.json').read_text())}
    originals={r['page']['page']:r['page']['original'] for r in history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for module in [assets,texture,history.fill,history.fill.frozen,history.fill.frozen.prior.prior]:deps[str(Path(module.__file__).relative_to(ROOT))]=sha(Path(module.__file__))
    for folder in [history.previous.OUT,history.extras.OUT,assets.OUT]:
        for name in ['manifest.json','complete.json']:deps[str((folder/name).relative_to(ROOT))]=sha(folder/name)
    for path in originals.values():deps[path]=sha(ROOT/path)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');current=None;results=[];poison=[]
    for row in rows:
        rid=row['id'];key=row['page']
        if current!=key:original=read(ROOT/originals[key]);current=key;gc.collect()
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        context=original[cy:cy+ch,cx:cx+cw];crop=context[s];lab=cv2.cvtColor(context.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        source=ROOT/row['baseline_local'];dest=OUT/'native'/rid
        base=read(source/'output.png');known=read(source/'reference_context.png',True);retained=read(source/'retained_applied_context.png',True)
        edit=read(source/'edit.png',True);applied=np.zeros_like(known);outputs={};stats={};asset=None
        if retained[s].any():
            asset,fields,base_field,old_seam,applied,regenerated=assets.color_assets(asset_rows.get(rid,row),lab,known,retained,base,s,old)
            distance=cv2.distanceTransform(applied.astype(np.uint8),cv2.DIST_L2,5);bw=np.exp(-(np.maximum(distance-1,0)/8.)**2)*applied;tw=np.clip((distance-1)/4.,0,1)*applied
            color=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*bw[:,:,None]
            combined=color.copy();combined[:,:,0]+=old_seam*tw
            assert np.array_equal(combined,base_field) and np.array_equal(history.fill.frozen.compose(combined,applied,base,s),base)
            strict=infer(lab,known,applied,24);assert np.array_equal(strict[0],old_seam)
            counts={};cache={24:strict}
            for size in SIZES:
                _,positions,total=donors(lab,known,size);counts[str(size)]=dict(total=total,selected=len(positions))
            adaptive=next((k for k in SIZES if counts[str(k)]['selected']>=4),None)
            dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'baseline_fields_Lab.npz',**fields)
            for name,item in dict(baseline_color_Lab=color,baseline_field_Lab=base_field,baseline_texture_seam=old_seam).items():np.save(dest/(name+'.npy'),item,allow_pickle=False)
            for path in asset.glob('*'):
                if path.is_file():deps[str(path.relative_to(ROOT))]=sha(path)
            for mode,size in [('uniform12',12 if counts['12']['selected']>=4 else None),('adaptive',adaptive)]:
                value=base.copy();changed=False
                if size is not None and size!=24:
                    if size not in cache:cache[size]=infer(lab,known,applied,size)
                    seam,average,sx,sy,donor_notes=cache[size];field=color.copy();field[:,:,0]+=seam*tw
                    assert np.array_equal(field[:,:,1:],base_field[:,:,1:])
                    update=applied&(tw>0);value=history.fill.frozen.compose(field,update,base,s);changed=True
                    md=dest/mode;md.mkdir(parents=True,exist_ok=True)
                    for name,item in dict(result_field_Lab=field,texture_seam=seam,texture_average=average,source_x=sx,source_y=sy).items():np.save(md/(name+'.npy'),item,allow_pickle=False)
                    dump(md/'donors.json',dict(context_xywh=row['context_xywh'],**donor_notes));write(md/'update_context.png',update)
                    if not any(v['id']==rid and v['patch_size']==size for v in poison):
                        bad=lab.copy();bad[~known]=(235,-180,130);repeat=infer(bad,known,applied,size)
                        for i in range(4):assert np.array_equal(cache[size][i],repeat[i])
                        assert donor_notes==repeat[4];poison.append(dict(id=rid,patch_size=size))
                assert not np.any(np.any(value!=base,axis=2)&~(applied&(tw>0))[s])
                outputs[mode]=value;stats[mode]=dict(selected_patch_size=size,texture_replaced=changed,
                    changed_vs_base=int(np.any(value!=base,axis=2).sum()),paper_applied=int(retained[s].sum()),
                    seam_L_std=float(np.std((cache[size][0] if changed else old_seam)[applied])) if applied.any() else None)
            dump(dest/'patch_counts.json',counts)
        else:
            for mode in MODES:outputs[mode]=base.copy();stats[mode]=dict(selected_patch_size=None,texture_replaced=False,changed_vs_base=0,paper_applied=int(retained[s].sum()),seam_L_std=None)
        for mode,value in outputs.items():
            assert not np.any(np.any(value!=base,axis=2)&~applied[s]) and not np.any(np.any(value!=crop,axis=2)&~edit)
            write(dest/mode/'output.png',value)
        for name,item in dict(original=crop,baseline=base,reference_context=known,retained_applied_context=retained,texture_applied_context=applied,edit=edit).items():write(dest/(name+'.png'),item)
        dump(dest/'color_source.json',dict(path=str(asset.relative_to(ROOT)) if asset else None,baseline_exactly_reconstructed=True))
        elsewhere='旧14ページ固定ROIの紙面外は保存DDRM。' if row['base_elsewhere']=='saved_DDRM' else '紙面外は原画像。'
        figure([crop,base]+[outputs[m] for m in MODES],['原画像','固定マスク・24px観測模様','同じ紙色・一律12px模様','同じ紙色・24→16→12参照確保'],OUT/'comparisons'/(rid+'.png'),
            row['title']+' · 参照パッチ尺度だけ',
            'マスク・紙面資格・適用・浮動小数a/b・紙色場は固定。元画像のL模様だけを移す。',
            elsewhere+' 紙面資格は緩めない。新しいDDRM・暖色輪郭・低次面なし。')
        results.append({**row,'texture_scale_diagnostics':stats});dump(OUT/'progress.json',dict(completed=len(results),total=77,latest=rid))
        print('Adaptive texture scale:',len(results),'/77',rid,{k:v['changed_vs_base'] for k,v in stats.items()},flush=True)
    dump(OUT/'final_dependencies.json',deps)
    for path,value in deps.items():assert sha(ROOT/path)==value
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',pages=25,ROIs=77,variants=154,baseline_field_and24_seam_exactly_reconstructed=True,
        reference_mask_applied_and_color_fixed=True,float_ab_exactly_unchanged=True,first_inside_pixel_unchanged=True,
        qualified_donor_pixels_only=True,all_nonreference_poison_cases=poison,original24_implementation_parity=True,
        outside_edit_changes=0,pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:adaptive observed paper texture scale,77 ROIs',flush=True)


if __name__=='__main__':main()
