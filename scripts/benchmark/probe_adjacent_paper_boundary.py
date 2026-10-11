"""Match adjacent known paper instead of extrapolating a bright-paper plane.

Preserve the first diagnostic, which increased the measured boundary mismatch.
Use identical fixed masks/DDRM outputs and all 22 cases, without new sampling.
"""
import csv
import gc
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import probe_paper_tone_boundary as previous
from build_npj_v6_comparison import dump,sha,read,write

ROOT=previous.ROOT
OUT=ROOT/'tmp_work/adjacent_paper_boundary_pilot_20261001'
DDRM=previous.DDRM
MASKS=previous.MASKS
MODES=previous.MODES
SETTINGS=dict(
    scope=previous.SETTINGS['scope'],mask_changes=False,DDRM_changes=False,
    original_in_edit_blending=False,paper_reference_gate=previous.SETTINGS['reference'],
    reference='adjacent unedited paper filtered by the same Lab/ink gates, no median-plane extrapolation',
    component_min_area=64,min_boundary_pairs=16,low_frequency_sigma_px=12,
    boundary_error_smoothing_sigma_px=8,boundary_scale_px=6,
    max_L_offset=6,max_ab_offset=4,
    tone='component median of known-paper minus generated adjacent pixel',
    boundary='smooth known-paper minus generated low-frequency boundary bias, fading inward',
    combined='tone first, then residual boundary bias',
    texture='no texture synthesis',pixel_GT=False,training=False,restoration_recomputed=False,
    development='second diagnostic after reviewing first; all 22 cases are development')


def boundary_samples(component,reference,original_lab,generated_lab,low_frequency):
    h,w=component.shape;sums=np.zeros((h,w,3),np.float32);counts=np.zeros((h,w),np.float32)
    pair_residuals=[]
    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        y0,y1=max(0,-dy),min(h,h-dy);x0,x1=max(0,-dx),min(w,w-dx)
        si=(slice(y0,y1),slice(x0,x1));so=(slice(y0+dy,y1+dy),slice(x0+dx,x1+dx))
        valid=component[si]&reference[so]
        if not valid.any():continue
        values=original_lab[so]
        sums[si]+=values*valid[:,:,None];counts[si]+=valid
        pair_residuals.append((values-generated_lab[si])[valid])
    if not pair_residuals:return None
    pairs=np.concatenate(pair_residuals)
    if len(pairs)<16:return None
    seeds=counts>0
    target=sums/np.maximum(counts[:,:,None],1)
    errors=(target-low_frequency)*seeds[:,:,None]
    weighted=cv2.GaussianBlur(counts,(0,0),8,borderType=cv2.BORDER_REFLECT_101)
    field=cv2.GaussianBlur(errors*counts[:,:,None],(0,0),8,borderType=cv2.BORDER_REFLECT_101)
    field/=np.maximum(weighted[:,:,None],1e-6)
    return np.median(pairs,axis=0),field,weighted,seeds,int(len(pairs))


def infer(original_context,reference_context,base,allowed,mask,offset):
    ox,oy=offset;h,w=base.shape[:2]
    assert not np.any(reference_context&mask)
    orig=previous.to_lab(original_context);orig=orig[oy:oy+h,ox:ox+w]
    ref=reference_context[oy:oy+h,ox:ox+w]
    lab=previous.to_lab(base);active=allowed&mask[oy:oy+h,ox:ox+w]
    n,labels,stats,_=cv2.connectedComponentsWithStats(active.astype(np.uint8),connectivity=8)
    modified={mode:lab.copy() for mode in MODES};adjusted=np.zeros((h,w),bool)
    components=[];seed_used=np.zeros((h,w),bool);limits=np.array([6,4,4],np.float32)
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA])
        if area<64:continue
        component=labels==i;lf=previous.smooth_generated(lab,component)
        sample=boundary_samples(component,ref,orig,lab,lf)
        if sample is None:
            components.append(dict(id=i,area=area,adjusted=False,reason='fewer than 16 known-paper boundary pairs'))
            continue
        bias,field,weights,seeds,count=sample
        bias=np.clip(bias,-limits,limits)
        distance=cv2.distanceTransform(component.astype(np.uint8),cv2.DIST_L2,5)
        weight=np.exp(-(np.maximum(distance-1,0)/6.)**2)*(weights>1e-4)
        border=np.clip(field,-limits,limits)*weight[:,:,None]
        residual=np.clip(field-bias,-limits,limits)*weight[:,:,None]
        modified['tone'][component]=(lab+bias)[component]
        modified['boundary'][component]=(lab+border)[component]
        modified['tone_boundary'][component]=(lab+bias+residual)[component]
        adjusted|=component;seed_used|=seeds
        components.append(dict(id=i,area=area,adjusted=True,boundary_pairs=count,
                               median_bias_Lab=bias.tolist()))
    outputs={}
    for mode,value in modified.items():
        image=base.copy();converted=previous.from_lab(value);image[adjusted]=converted[adjusted]
        assert not np.any(np.any(image!=base,axis=2)&~active)
        outputs[mode]=image
    return outputs,adjusted,seed_used,components


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dependencies={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
        str(Path(previous.__file__).relative_to(ROOT)):sha(Path(previous.__file__))}
    for folder in [previous.OUT,DDRM]:
        for name in ['manifest.json','complete.json']:dependencies[str((folder/name).relative_to(ROOT))]=sha(folder/name)
        for rel,digest in json.loads((folder/'manifest.json').read_text())['dependencies'].items():assert sha(ROOT/rel)==digest,rel
        for rel,digest in json.loads((folder/'complete.json').read_text())['outputs'].items():assert sha(folder/rel)==digest,rel
    manifest=dict(settings=SETTINGS,dependencies=dependencies,source_sha256=sha(Path(__file__)),
        pages=14,regions=22,selection='same all 22 development ROIs; no outcome omission')
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=[];poison_checked=[]
    first_rows={r['id']:r for r in json.loads((previous.OUT/'diagnostics.json').read_text())}
    for source in previous.margin_probe.SOURCES:
        for page in json.loads((source/'baseline_review.json').read_text())['cases']:
            idpage=page['page'];original=read(ROOT/page['original']);H,W=original.shape[:2]
            edit=read(MASKS/'candidate_pages'/idpage/'dark_support/edit.png',True)
            band=read(MASKS/'candidate_pages'/idpage/'features/top_margin.png',True)
            protected=np.zeros((H,W),bool)
            for key in ['protect','hold','red','extra']:protected|=read(source/'v6_masks'/idpage/(key+'.png'),True)
            protected&=~edit
            _,reference=previous.clean_reference(original,edit,protected,band)
            for roi in page['rois']:
                id=roi['id'];x,y,w,h=roi['xywh'];x0,y0=max(0,x-64),max(0,y-64);x1,y1=min(W,x+w+64),min(H,y+h+64)
                context=original[y0:y1,x0:x1];mask=edit[y0:y1,x0:x1];ref=reference[y0:y1,x0:x1]
                offset=(x-x0,y-y0);s=(slice(y-y0,y-y0+h),slice(x-x0,x-x0+w));crop=context[s]
                native=DDRM/'native'/id;base=read(native/'candidate.png');v6=read(native/'v6.png')
                assert np.array_equal(crop,read(native/'original.png'))
                outputs,adjusted,seeds,components=infer(context,ref,base,band[y:y+h,x:x+w],mask,offset)
                if id in ['hole_edge_new','new_margin_hole','new_brown_stamp']:
                    poison=context.copy();poison[mask]=(219,31,73)
                    repeat=infer(poison,ref,base,band[y:y+h,x:x+w],mask,offset)[0]
                    for mode in MODES:assert np.array_equal(outputs[mode],repeat[mode])
                    poison_checked.append(id)
                dest=OUT/'native'/id
                for name,value in dict(original=crop,v6=v6,base=base,edit=edit[y:y+h,x:x+w],adjusted=adjusted,
                    boundary_seeds=seeds,reference_context=ref,**outputs).items():
                    write(dest/(name+'.png'),value);assert np.array_equal(read(dest/(name+'.png'),value.dtype==bool),value)
                dump(dest/'components.json',components)
                for value in outputs.values():
                    assert not np.any(np.any(value!=crop,axis=2)&~edit[y:y+h,x:x+w])
                # Evaluate every method on the identical first diagnostic's pair set.
                evalmask=read(previous.OUT/'native'/id/'adjusted.png',True)
                diagnostics={}
                for mode,image in dict(base=base,plane=read(previous.OUT/'native'/id/'tone_boundary.png'),**outputs).items():
                    lab=previous.to_lab(image);jumps=[];localref=ref[s]
                    for axis in [0,1]:
                        if axis==0:
                            pairs=(evalmask[:-1]&localref[1:])|(evalmask[1:]&localref[:-1]);delta=lab[:-1]-lab[1:]
                        else:
                            pairs=(evalmask[:,:-1]&localref[:,1:])|(evalmask[:,1:]&localref[:,:-1]);delta=lab[:,:-1]-lab[:,1:]
                        if pairs.any():jumps.append(np.linalg.norm(delta[pairs],axis=1))
                    values=np.concatenate(jumps) if jumps else np.empty(0)
                    diagnostics[mode]=dict(boundary_pairs=int(len(values)),
                        median_known_paper_boundary_deltaLab=float(np.median(values)) if len(values) else None,
                        changed_vs_base=int(np.any(image!=base,axis=2).sum()))
                first=first_rows[id]
                assert diagnostics['base']['boundary_pairs']==first['diagnostics']['base']['boundary_pairs']
                assert diagnostics['base']['median_known_paper_boundary_deltaLab']==first['diagnostics']['base']['median_known_paper_boundary_deltaLab']
                note=f'合成補正対象 {int(adjusted.sum()):,}画素。隣接する未編集紙面だけを補正の参照に使います。'
                previous.figure([crop,v6,base,outputs['tone_boundary']],['原画像','固定v6 → DDRM','現在の余白試作','隣接紙面で合成補正'],
                    OUT/'comparisons'/(id+'.png'),roi['title']+' · '+idpage,note)
                previous.figure([crop,base,outputs['tone'],outputs['boundary'],outputs['tone_boundary']],
                    ['原画像','補正前','隣接紙色のみ','隣接境界のみ','隣接紙色＋境界'],
                    OUT/'ablation'/(id+'.png'),roi['title']+' · 隣接紙面の要素比較',note)
                previous.figure([crop,base,read(previous.OUT/'native'/id/'tone_boundary.png'),outputs['tone_boundary']],
                    ['原画像','補正前','明るい紙の平面で補正','隣接紙面で補正'],
                    OUT/'reference_ablation'/(id+'.png'),roi['title']+' · 紙色参照の違い',note)
                distance=cv2.distanceTransform(evalmask.astype(np.uint8),cv2.DIST_L2,5);core=evalmask&(distance>=4)
                texture={}
                for mode,image in dict(base=base,**outputs).items():
                    light=previous.to_lab(image)[:,:,0];hp=light-cv2.GaussianBlur(light,(0,0),1.2)
                    texture[mode+'_L_highpass_std']=float(np.std(hp[core])) if core.any() else None
                rows.append(dict(id=id,page=idpage,title=roi['title'],xywh=roi['xywh'],
                    adjusted_pixels=int(adjusted.sum()),components=components,diagnostics=diagnostics,
                    texture=texture,outside_edit_changed_pixels=0))
                print('Adjacent boundary:',len(rows),'/22',id,'adjusted',int(adjusted.sum()),flush=True)
            del original,edit,band,protected,reference;gc.collect()
    dump(OUT/'diagnostics.json',rows)
    with (OUT/'diagnostics.csv').open('w') as stream:
        flat=[dict(id=r['id'],adjusted_pixels=r['adjusted_pixels'],
            **{m+'_boundary_deltaLab':v['median_known_paper_boundary_deltaLab'] for m,v in r['diagnostics'].items()},
            **r['texture']) for r in rows]
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    for rel,digest in dependencies.items():assert sha(ROOT/rel)==digest,rel
    dump(OUT/'verification.json',dict(status='passed',pages=14,regions=22,mask_and_DDRM_fixed=True,
        original_damage_poison_invariance=poison_checked,native_pixels=True,
        outside_edit_changed_pixels=0,boundary_metrics_identical_pair_set=True,
        source_and_output_hashes=True,texture_synthesis=False,pixel_GT=False,DDRM_recomputed=False))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*')
        if p.is_file() and p.suffix in ['.png','.json','.csv'] and p.name not in ['complete.json','manifest.json']}))
    print('Adjacent paper boundary complete: 22 ROIs',flush=True)


if __name__=='__main__':main()
