"""Postprocess fixed saved DDRM outputs; separate paper tone and boundary bias.

Fit local paper planes from unedited, conservatively filtered pixels. Every
variant keeps the same edit mask and never blends original damaged pixels back
into it. This local output experiment does not re-run or train the diffusion.
"""
import csv
import gc
import json
import os
from pathlib import Path
import shutil

os.environ['OPENBLAS_NUM_THREADS']='4'
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import probe_paper_margin_defects as margin_probe
from build_npj_v6_comparison import dump,sha,read,write,FONT_PATH

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp_work/paper_tone_boundary_pilot_20261001'
DDRM=ROOT/'tmp_work/paper_margin_defect_ddrm_20261001'
MASKS=margin_probe.OUT
MODES=['tone','boundary','tone_boundary']
SETTINGS=dict(
    scope='saved local DDRM output, only current edit intersect fixed top-margin band',
    mask_changes=False,DDRM_changes=False,original_in_edit_blending=False,
    reference=dict(context_px=64,ring_inner_px=3,ring_outer_px=48,min_pixels=256,
        min_L=50,max_paper_L_drop=4,min_paper_L_drop=-6,max_abs_delta_a=2,
        max_abs_delta_b=4,known_dark_guard_px=4,max_gradient=3,
        max_samples=10000,robust_plane_iterations=3),
    component_min_area=64,low_frequency_sigma_px=12,
    max_L_offset=6,max_ab_offset=4,boundary_scale_px=6,
    tone='component-wise median Lab bias to robust plane fitted from known paper',
    boundary='smooth low-frequency Lab bias at the interior boundary, fading inward',
    combined='tone bias, then boundary residual; no double correction',
    texture='diagnostic only; no new texture or random noise added',
    evaluation='all 22 fixed ROIs; consistency proxies, not GT accuracy',
    pixel_GT=False,training=False,restoration_recomputed=False)
cv2.setNumThreads(1)


def to_lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)


def from_lab(lab):
    bgr=cv2.cvtColor(lab.astype(np.float32),cv2.COLOR_LAB2BGR)
    return np.rint(np.clip(bgr,0,1)*255).astype(np.uint8)


def clean_reference(original,edit,protected,margin):
    lab=to_lab(cv2.medianBlur(original,3))
    paper=margin_probe.first.paper_estimate(lab)
    drop=paper[:,:,0]-lab[:,:,0]
    plain=(drop<=4)&(drop>=-6)&(lab[:,:,0]>=50)
    plain&=(np.abs(lab[:,:,1]-paper[:,:,1])<=2)&(np.abs(lab[:,:,2]-paper[:,:,2])<=4)
    base=plain&~edit&~protected&margin
    guard=cv2.dilate(((drop>6)|protected).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
    light=cv2.GaussianBlur(lab[:,:,0],(0,0),1)
    gx=cv2.Sobel(light,cv2.CV_32F,1,0,ksize=3,scale=.125)
    gy=cv2.Sobel(light,cv2.CV_32F,0,1,ksize=3,scale=.125)
    return base&~guard&(np.hypot(gx,gy)<=3),base


def robust_plane(original_lab,reference,cx,cy):
    ys,xs=np.nonzero(reference)
    if len(xs)<256:return None
    step=max(1,int(np.ceil(len(xs)/10000)));xs=xs[::step];ys=ys[::step]
    design=np.stack([np.ones(len(xs)),(xs-cx)/48.,(ys-cy)/48.],axis=1)
    if np.linalg.cond(design)>1e4:return None
    target=original_lab[ys,xs].astype(np.float64);weights=np.ones(len(xs))
    for _ in range(3):
        root=np.sqrt(weights)
        coeff=np.linalg.lstsq(design*root[:,None],target*root[:,None],rcond=None)[0]
        residual=np.linalg.norm(target-design@coeff,axis=1)
        scale=max(.5,float(np.median(residual))*1.4826)
        weights=np.minimum(1,2*scale/np.maximum(residual,1e-6))
    return coeff.astype(np.float32)


def smooth_generated(lab,component):
    weights=cv2.GaussianBlur(component.astype(np.float32),(0,0),12,borderType=cv2.BORDER_REFLECT_101)
    numer=cv2.GaussianBlur(lab*component[:,:,None],(0,0),12,borderType=cv2.BORDER_REFLECT_101)
    return numer/np.maximum(weights[:,:,None],1e-6)


def postprocess(original_context,reference_context,metric_reference,base,allowed,mask,offset):
    """Only reference_context=True positions read original values for correction.

    Original pixels in the edit mask may be replaced by arbitrary poison values
    with references frozen; outputs must be identical. Base is the saved DDRM.
    """
    assert not np.any(reference_context&mask)
    ox,oy=offset;h,w=base.shape[:2];local_orig_lab=to_lab(original_context)
    base_lab=to_lab(base);modified={m:base_lab.copy() for m in MODES}
    active=allowed&mask[oy:oy+h,ox:ox+w]
    n,labels,stats,_=cv2.connectedComponentsWithStats(active.astype(np.uint8),connectivity=8)
    adjusted=np.zeros((h,w),bool);ref_used=np.zeros(reference_context.shape,bool)
    components=[];limits=np.array([6,4,4],np.float32)
    yy,xx=np.indices((h,w),dtype=np.float32)
    for i in range(1,n):
        x,y,rw,rh,area=map(int,stats[i])
        if area<64:continue
        component=labels==i
        contextual=np.zeros(reference_context.shape,np.uint8)
        contextual[oy:oy+h,ox:ox+w]=component.astype(np.uint8)
        outward=cv2.distanceTransform(1-contextual,cv2.DIST_L2,5)
        ring=reference_context&(outward>=3)&(outward<=48)
        cy,cx=np.mean(np.argwhere(contextual>0),axis=0)
        coeff=robust_plane(local_orig_lab,ring,cx,cy)
        if coeff is None:
            components.append(dict(id=i,area=area,adjusted=False,reference_pixels=int(ring.sum())))
            continue
        plane=(coeff[0]+((xx+ox-cx)/48.)[:,:,None]*coeff[1]
               +((yy+oy-cy)/48.)[:,:,None]*coeff[2])
        lf=smooth_generated(base_lab,component)
        inner=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
        core=component&(inner>=4)
        if core.sum()<16:core=component
        bias=np.clip(np.median((plane-base_lab)[core],axis=0),-limits,limits)
        weight=np.exp(-((np.maximum(inner-1,0)/6.)**2)).astype(np.float32)
        border_bias=np.clip(plane-lf,-limits,limits)*weight[:,:,None]
        residual_bias=np.clip(plane-(lf+bias),-limits,limits)*weight[:,:,None]
        modified['tone'][component]=(base_lab+bias)[component]
        modified['boundary'][component]=(base_lab+border_bias)[component]
        modified['tone_boundary'][component]=(base_lab+bias+residual_bias)[component]
        adjusted|=component;ref_used|=ring
        components.append(dict(id=i,area=area,adjusted=True,reference_pixels=int(ring.sum()),
            core_pixels=int(core.sum()),median_bias_Lab=bias.tolist(),
            paper_plane_Lab_coefficients=coeff.tolist()))
    variants={}
    for mode,lab in modified.items():
        value=base.copy();converted=from_lab(lab);value[adjusted]=converted[adjusted]
        assert not np.any(np.any(value!=base,axis=2)&~active)
        variants[mode]=value
    local_reference=metric_reference[oy:oy+h,ox:ox+w]
    diagnostic={}
    for name,value in dict(base=base,**variants).items():
        lab=to_lab(value);jumps=[]
        for axis in [0,1]:
            if axis==0:
                a,b=adjusted[:-1],adjusted[1:];ra,rb=local_reference[:-1],local_reference[1:]
                pairs=(a&rb)|(b&ra);diff=lab[:-1]-lab[1:]
            else:
                a,b=adjusted[:,:-1],adjusted[:,1:];ra,rb=local_reference[:,:-1],local_reference[:,1:]
                pairs=(a&rb)|(b&ra);diff=lab[:,:-1]-lab[:,1:]
            if pairs.any():jumps.append(np.linalg.norm(diff[pairs],axis=1))
        values=np.concatenate(jumps) if jumps else np.empty(0)
        diagnostic[name]=dict(boundary_pairs=int(len(values)),
            median_known_paper_boundary_deltaLab=float(np.median(values)) if len(values) else None,
            changed_vs_base=int(np.any(value!=base,axis=2).sum()))
    # Texture consistency is a diagnostic; no reference texture is synthesized.
    distance=cv2.distanceTransform(adjusted.astype(np.uint8),cv2.DIST_L2,5)
    core=adjusted&(distance>=4)
    orig_light=local_orig_lab[:,:,0]
    original_hp=orig_light-cv2.GaussianBlur(orig_light,(0,0),1.2)
    texture_reference=ref_used&(cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)>=4)
    texture=dict(reference_pixels=int(texture_reference.sum()),edit_core_pixels=int(core.sum()),
        reference_L_highpass_std=float(np.std(original_hp[texture_reference])) if texture_reference.any() else None)
    for name,value in dict(base=base,**variants).items():
        light=to_lab(value)[:,:,0];hp=light-cv2.GaussianBlur(light,(0,0),1.2)
        texture[name+'_L_highpass_std']=float(np.std(hp[core])) if core.any() else None
    return variants,adjusted,ref_used,components,diagnostic,texture


def figure(images,labels,path,title,note):
    h,w=images[0].shape[:2];scale=min(480/w,900/h);dw,dh=round(w*scale),round(h*scale)
    canvas=Image.new('RGB',(len(images)*(dw+20)+28,dh+240),(250,248,243));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT_PATH,24);small=ImageFont.truetype(FONT_PATH,20)
    draw.text((24,14),title,font=font,fill='black')
    draw.text((24,51),'保存済みDDRMと編集マスクを固定。周囲の未編集紙面を使う後処理の比較。',font=small,fill='black')
    for i,(im,label) in enumerate(zip(images,labels)):
        x=24+i*(dw+20);draw.text((x,91),label,font=font,fill='black')
        resized=cv2.resize(im,(dw,dh),interpolation=cv2.INTER_NEAREST)
        canvas.paste(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)),(x,128))
    draw.text((24,dh+147),'マスク拡張・拡散再実行・原画像の損傷画素の混合は行っていません。',font=small,fill='black')
    draw.text((24,dh+181),note,font=small,fill='black');path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        str(Path(margin_probe.__file__).relative_to(ROOT)):sha(Path(margin_probe.__file__)),
        str(Path(margin_probe.first.__file__).relative_to(ROOT)):sha(Path(margin_probe.first.__file__))}
    for root,mname,cname in [(DDRM,'manifest.json','complete.json'),(MASKS,'candidate_manifest.json','candidate_complete.json')]:
        for name in [mname,cname]:dependencies[str((root/name).relative_to(ROOT))]=sha(root/name)
        for rel,digest in json.loads((root/mname).read_text())['dependencies'].items():assert sha(ROOT/rel)==digest,rel
        for rel,digest in json.loads((root/cname).read_text())['outputs'].items():assert sha(root/rel)==digest,rel
    cases=[]
    for source in margin_probe.SOURCES:
        p=source/'baseline_review.json';dependencies[str(p.relative_to(ROOT))]=sha(p)
        for row in json.loads(p.read_text())['cases']:
            dependencies[row['original']]=sha(ROOT/row['original']);cases.append((source,row))
            for key in ['edit','protect','hold','red','extra']:
                p=source/'v6_masks'/row['page']/(key+'.png');dependencies[str(p.relative_to(ROOT))]=sha(p)
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)),
        pages=14,regions=22,selection='all previous 22 ROIs, no outcome selection',
        provenance='postprocessing the fixed saved paper-margin DDRM; not a new diffusion method')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest,'Use a new version for changed code'
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py')
    rows=[];poison_checked=[]
    for source,page in cases:
        idpage=page['page'];original=read(ROOT/page['original']);H,W=original.shape[:2]
        edit=read(MASKS/'candidate_pages'/idpage/'dark_support/edit.png',True)
        band=read(MASKS/'candidate_pages'/idpage/'features/top_margin.png',True)
        protected=np.zeros((H,W),bool)
        for key in ['protect','hold','red','extra']:protected|=read(source/'v6_masks'/idpage/(key+'.png'),True)
        protected&=~edit
        reference,metric_reference=clean_reference(original,edit,protected,band)
        write(OUT/'reference_pages'/idpage/'clean_paper.png',reference)
        for roi in page['rois']:
            id=roi['id'];x,y,w,h=roi['xywh'];x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
            context=original[y0:y1,x0:x1];s=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w));crop=context[s]
            native=DDRM/'native'/id;base=read(native/'candidate.png');v6=read(native/'v6.png')
            assert np.array_equal(crop,read(native/'original.png'))
            assert np.array_equal(edit[y:y+h,x:x+w],read(native/'candidate_edit.png',True))
            local_mask=edit[y0:y1,x0:x1];allowed=band[y:y+h,x:x+w]
            ref=reference[y0:y1,x0:x1];metric=metric_reference[y0:y1,x0:x1]
            variants,adjusted,used,components,diagnostic,texture=postprocess(context,ref,metric,base,allowed,local_mask,(x-x0,y-y0))
            if id in ['hole_edge_new','new_margin_hole','new_brown_stamp']:
                poison=context.copy();poison[local_mask]=(17,83,241)
                repeat=postprocess(poison,ref,metric,base,allowed,local_mask,(x-x0,y-y0))[0]
                for mode in MODES:assert np.array_equal(variants[mode],repeat[mode])
                poison_checked.append(id)
            dest=OUT/'native'/id
            for name,value in dict(original=crop,v6=v6,base=base,edit=edit[y:y+h,x:x+w],adjusted=adjusted,**variants).items():
                write(dest/(name+'.png'),value);assert np.array_equal(read(dest/(name+'.png'),value.dtype==bool),value)
            write(dest/'reference_context.png',used)
            dump(dest/'components.json',components)
            for mode,value in variants.items():
                assert not np.any(np.any(value!=base,axis=2)&~edit[y:y+h,x:x+w])
                assert not np.any(np.any(value!=crop,axis=2)&~edit[y:y+h,x:x+w])
            note=f'合成補正対象 {int(adjusted.sum()):,}画素。編集対象の追加なし。紙の質感は今回生成していません。'
            figure([crop,v6,base,variants['tone_boundary']],['原画像','固定v6 → DDRM','現在の余白試作','紙色＋境界の補正'],
                   OUT/'comparisons'/(id+'.png'),roi['title']+' · '+idpage,note)
            figure([crop,base,variants['tone'],variants['boundary'],variants['tone_boundary']],
                   ['原画像','補正前','紙色のみ','境界のみ','紙色＋境界'],
                   OUT/'ablation'/(id+'.png'),roi['title']+' · 後処理の要素比較',note)
            rows.append(dict(id=id,page=idpage,xywh=roi['xywh'],title=roi['title'],baseline_failure=roi['baseline_failure'],
                adjusted_pixels=int(adjusted.sum()),reference_pixels=int(used.sum()),components=components,
                diagnostics=diagnostic,texture=texture,outside_edit_changed_pixels=0))
            print('Tone/boundary:',len(rows),'/22',id,'adjusted',int(adjusted.sum()),flush=True)
        del original,edit,band,protected,reference,metric_reference;gc.collect()
    dump(OUT/'diagnostics.json',rows)
    table=[]
    for row in rows:
        table.append(dict(id=row['id'],page=row['page'],adjusted_pixels=row['adjusted_pixels'],
            plane_reference_pixels=row['reference_pixels'],
            **{mode+'_boundary_deltaLab':v['median_known_paper_boundary_deltaLab'] for mode,v in row['diagnostics'].items()},
            **row['texture']))
    with (OUT/'diagnostics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=14,regions=22,mask_and_DDRM_output_fixed=True,
        original_damage_poison_invariance=poison_checked,reference_excludes_edit=True,
        native_pixels=True,outside_edit_changed_pixels=0,source_and_output_hashes=True,
        texture_synthesis=False,pixel_GT=False,DDRM_recomputed=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.csv'] and p.name not in ['complete.json','manifest.json']}))
    print('Paper tone and boundary comparison complete: 22 ROIs',flush=True)


if __name__=='__main__':main()
