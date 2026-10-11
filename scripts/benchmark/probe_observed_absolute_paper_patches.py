"""Compare absolute observed paper L/Lab, with the existing applied mask fixed."""
import gc
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import diagnose_paper_boundary_reference_bias as diagnostic
import probe_paper_boundary_pair_gate_v2 as fill
import probe_paper_seam_texture as seams
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=diagnostic.ROOT
OUT=ROOT/'tmp_work/observed_absolute_paper_patches_pilot_20261001'
MODES=['observed_L','observed_Lab']
SETTINGS=dict(pages=29,ROIs=90,baseline='directional abstention77 + new4-books13',
    fixed='current effective applied geometry, directional holds, original known-paper qualification',
    donor=dict(patch_px=24,stride_px=8,max_candidates=512,min_candidates=4,
        qualification='every pixel in the frozen median-qualified known-paper mask',
        signal='absolute original float Lab; no mean subtraction, clipping, offsets or noise'),
    quilting=dict(overlap_px=8,tile_stride_px=16,first_patch='nearest full-Lab mean to median known Lab within64px of component',
        cost='mean squared Lab overlap error plus .02 times reuse count',
        seam='minimum-error left and top cuts, retain old on either keep-old seam side'),
    matched='identical donor placement and raw source coordinates for observed_L and observed_Lab',
    blend='baseline + (donor - baseline) * [1-exp(-(max(distance-1,0)/8)^2)]; applied only',
    observed_L='replace only baseline float L with blended absolute donor L',
    observed_Lab='replace all three channels with the same blended absolute donor Lab',
    fallback='fewer than4 qualified24px patches: retain entire baseline; no gate/scale relaxation',
    qualification_caveat='median qualification may admit individual raw outliers; not ink/damage ground truth',
    colour_corrections_combined=False,new_DDRM=False,pixel_GT=False,development=True)


def donors(lab,known):
    h,w=known.shape;integral=cv2.integral(known.astype(np.uint8));positions=[]
    for y in range(0,h-23,8):
        for x in range(0,w-23,8):
            if integral[y+24,x+24]-integral[y,x+24]-integral[y+24,x]+integral[y,x]==576:positions.append((x,y))
    if len(positions)>512:positions=[positions[i] for i in np.linspace(0,len(positions)-1,512,dtype=int)]
    if len(positions)<4:return None,positions
    return np.stack([lab[y:y+24,x:x+24].copy() for x,y in positions]),positions


def quilt(patches,positions,shape,target):
    h,w=shape;H,W=max(h,24)+24,max(w,24)+24
    image=np.zeros((H,W,3),np.float32);occupied=np.zeros((H,W),bool)
    sx=np.full((H,W),-1,np.int32);sy=sx.copy();use=np.zeros(len(patches),np.int32);records=[]
    first=int(np.argmin(np.mean((np.mean(patches,axis=(1,2))-target)**2,axis=1)))
    yy,xx=np.indices((24,24))
    for y in range(0,max(h,24),16):
        for x in range(0,max(w,24),16):
            old=image[y:y+24,x:x+24];overlap=occupied[y:y+24,x:x+24]
            if overlap.any():
                cost=np.mean((patches[:,overlap]-old[overlap])**2,axis=(1,2))+.02*use
                chosen=int(np.argmin(cost))
            else:chosen=first
            patch=patches[chosen];error=np.mean((old-patch)**2,axis=2);take=np.ones((24,24),bool)
            if x>0:
                path=seams.minimum_path(error[:,:8]);take[:,:8]&=np.arange(8)[None,:]>path[:,None]
            if y>0:
                path=seams.minimum_path(error[:8,:].T);take[:8,:]&=np.arange(8)[:,None]>path[None,:]
            take|=~overlap;old[take]=patch[take];occupied[y:y+24,x:x+24]=True
            px,py=positions[chosen];sx[y:y+24,x:x+24][take]=(xx+px)[take];sy[y:y+24,x:x+24][take]=(yy+py)[take]
            use[chosen]+=1;records.append(dict(x=x,y=y,donor=chosen,source_xy=[px,py]))
    assert (sx[:h,:w]>=0).all()
    return image[:h,:w],sx[:h,:w],sy[:h,:w],records


def infer(lab,known,app,baseline):
    patches,positions=donors(lab,known);q=np.zeros_like(lab)
    sx=np.full(app.shape,-1,np.int32);sy=sx.copy();records=[]
    if patches is not None:
        n,labels,stats,_=cv2.connectedComponentsWithStats(app.astype(np.uint8),connectivity=4)
        for k in range(1,n):
            x,y,w,h,area=map(int,stats[k]);component=labels==k
            distance=cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5)
            ring=known&(distance<=64);assert ring.any()
            target=np.median(lab[ring],axis=0)
            image,px,py,tiles=quilt(patches,positions,(h,w),target)
            local=component[y:y+h,x:x+w]
            q[y:y+h,x:x+w][local]=image[local];sx[y:y+h,x:x+w][local]=px[local];sy[y:y+h,x:x+w][local]=py[local]
            records.append(dict(id=k,xywh=[x,y,w,h],area=area,ring_pixels=int(ring.sum()),target_Lab=target.tolist(),tiles=tiles))
    distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)
    alpha=(1-np.exp(-(np.maximum(distance-1,0)/8.)**2))*app*(sx>=0)
    fields={m:baseline.copy() for m in MODES}
    fields['observed_L'][:,:,0]+=(q[:,:,0]-baseline[:,:,0])*alpha
    fields['observed_Lab']+=(q-baseline)*alpha[:,:,None]
    return fields,positions,records,q,sx,sy,alpha


def metrics(image,base,app,known,s):
    lab=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_BGR2LAB);L=lab[:,:,0]
    core=app[s]&(cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)[s]>=6)
    hp=L-cv2.GaussianBlur(L,(0,0),1.2);target=app[s];ref=known[s];jumps=[]
    for a,b,ra,rb,delta in [(target[:-1],target[1:],ref[:-1],ref[1:],lab[:-1]-lab[1:]),
        (target[:,:-1],target[:,1:],ref[:,:-1],ref[:,1:],lab[:,:-1]-lab[:,1:])]:
        pairs=(a&rb)|(b&ra)
        if pairs.any():jumps.append(np.linalg.norm(delta[pairs],axis=1))
    j=np.concatenate(jumps) if jumps else np.empty(0)
    return dict(changed_vs_base=int(np.any(image!=base,axis=2).sum()),core_L_highpass_std=float(np.std(hp[core])) if core.any() else None,
        boundary_pairs=len(j),median_boundary_deltaLab=float(np.median(j)) if len(j) else None,
        applied_L_le_80=int(((L<=80)&target).sum()))


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(diagnostic.OUT);check_bundle(diagnostic.previous.texture.OUT)
    rows=json.loads((diagnostic.OUT/'diagnostics.json').read_text())['regions']
    originals={r['page']['page']:r['page']['original'] for r in diagnostic.previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    originals.update({r['page']:r['original'] for r in json.loads((diagnostic.fresh.INPUT/'baseline_review.json').read_text())['cases']})
    deps={str(Path(p).relative_to(ROOT)):sha(Path(p)) for p in [__file__,seams.__file__,fill.__file__]}
    for folder in [diagnostic.OUT,diagnostic.previous.texture.OUT]:
        for n in ['manifest.json','complete.json']:deps[str((folder/n).relative_to(ROOT))]=sha(folder/n)
    for row in rows:
        source=ROOT/row['local']
        for name in [row['output_file'],row['applied_file'],'reference_context.png']:
            p=source/name;deps[str(p.relative_to(ROOT))]=sha(p)
    for key in {r['page'] for r in rows}:deps[originals[key]]=sha(ROOT/originals[key])
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');results=[];poison=[];current=None
    for index,row in enumerate(rows,1):
        rid=row['id'];key=row['page']
        if current!=key:
            original=read(ROOT/originals[key]);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);current=key;gc.collect()
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=full_lab[cs];crop=original[y:y+h,x:x+w];source=ROOT/row['local'];base=read(source/row['output_file'])
        app=read(source/row['applied_file'],True);known=read(source/'reference_context.png',True);dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True)
        assert not np.any(known&app)
        outputs={};stats={};field_path=None;positions=[];notes=[]
        if app[s].any():
            field_path=diagnostic.previous.texture.OUT/'native'/rid/'baseline_field_Lab.npy'
            field=np.load(field_path,allow_pickle=False);assert np.array_equal(fill.frozen.compose(field,app,base,s),base)
            fields,positions,notes,q,sx,sy,alpha=infer(lab,known,app,field)
            bad=lab.copy();bad[~known]=(240,-210,140);repeat=infer(bad,known,app,field)
            assert positions==repeat[1] and notes==repeat[2]
            for m in MODES:assert np.array_equal(fields[m],repeat[0][m])
            for a,b in zip([q,sx,sy,alpha],repeat[3:]):assert np.array_equal(a,b)
            poison.append(rid);selected=sx>=0
            assert not np.any(selected&~app) and known[sy[selected],sx[selected]].all()
            assert np.array_equal(q[selected],lab[sy[selected],sx[selected]])
            update=alpha>0
            for n,a in dict(baseline_field_Lab=field,quilt_Lab=q,source_x=sx,source_y=sy,alpha=alpha).items():np.save(dest/(n+'.npy'),a,allow_pickle=False)
            for mode in MODES:
                outputs[mode]=fill.frozen.compose(fields[mode],update,base,s)
                np.save(dest/(mode+'_field_Lab.npy'),fields[mode],allow_pickle=False)
                assert np.array_equal(outputs[mode][~update[s]],base[~update[s]])
        else:
            update=np.zeros_like(app)
            outputs={m:base.copy() for m in MODES}
        for n,a in dict(original=crop,baseline=base,applied_context=app,reference_context=known,update_context=update,**outputs).items():write(dest/(n+'.png'),a)
        dump(dest/'donors.json',dict(patch_xy=positions,components=notes))
        for mode,image in dict(baseline=base,**outputs).items():stats[mode]=metrics(image,base,app,known,s)
        figure([crop,base]+[outputs[m] for m in MODES],['原画像','現行補修・方向保留','観測紙パッチの絶対L','同じパッチの絶対Lab'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 紙の観測値を移す',
            f'24px角の全画素が従来の紙参照内。候補{len(positions)}個。先頭境界は前回を維持、内部へ8pxで移行。',
            '同じマスク・同じパッチ配置。色補正併用なし。元の中央値資格は画素GTではない。')
        results.append({**row,'original_source':originals[key],'field_source':str(field_path.relative_to(ROOT)) if field_path else None,
            'donor_patches':len(positions),'applied_ROI':int(app[s].sum()),'update_ROI':int(update[s].sum()),'patch_diagnostics':stats})
        dump(OUT/'progress.json',dict(completed=index,total=90,latest=rid));print('Absolute observed paper:',index,'/90',rid,len(positions),[(m,stats[m]['changed_vs_base']) for m in MODES],flush=True)
    for n,h in deps.items():assert sha(ROOT/n)==h,n
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,nonknown_poison_invariance=poison,baseline_recomposition=True,
        source_values_exact_raw_Lab=True,outside_update_exact_baseline=True,no_new_applied_pixels=True,qualification='old median-qualified known paper',pixel_GT=False,new_DDRM=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete:absolute observed24px paper patch trial90 ROIs',flush=True)


if __name__=='__main__':main()
