"""Independently reconstruct component gates and scalar-colour composition."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from build_npj_v6_comparison import ROOT, read, sha, dump
from package_saved_npj_inference_v6_review import check

PILOT=ROOT/'tmp_work/npj_docres_gap_gate_pilot_20261002'
REVIEW=ROOT/'tmp_work/npj_docres_gap_gate_review_20261002'
TRIAL=ROOT/'tmp_work/npj_literature_trial_20261002'
OUT=ROOT/'tmp_work/npj_docres_gap_gate_independent_audit_20261002'


def main():
    assert not (OUT/'complete.json').exists()
    manifest=check(PILOT);records=[];total_panels=0
    for spec in manifest['frames']:
        fid=spec['id'];src=PILOT/'native'/fid
        original=read(ROOT/spec['input']);base=read(ROOT/spec['saved_npj'])
        assert np.array_equal(original,read(src/'saved_input.png'))
        assert np.array_equal(base,read(src/'saved_npj.png'))
        masks={k:read(src/(k+'_mask.png'),True) for k in ['ungated','gate_05','gate_09','red','hold','extra']}
        role_dir=TRIAL/'inputs'/fid
        roles={k:read(role_dir/(k+'.png'),True) for k in ['protect','hold','edit','red','extra']}
        method='docres' if spec['input_reference_exact'] else 'matched_docres'
        check(TRIAL/method)
        p=np.load(TRIAL/method/fid/'background_probability.npy',allow_pickle=False)
        n,l,stats,_=cv2.connectedComponentsWithStats(masks['ungated'].astype(np.uint8),connectivity=8)
        lookup={tuple(map(int,s)):i for i,s in enumerate(stats[1:],1)}
        assert len(lookup)==n-1
        diagnostics=json.loads((src/'diagnostics.json').read_text())
        assert n-1==len(diagnostics['components'])
        expected={k:np.zeros(l.shape,bool) for k in ['gate_05','gate_09']}
        for c in diagnostics['components']:
            i=lookup[tuple(c['xywh']+[c['pixels']])]
            x,y,w,h=c['xywh'];component=l[y:y+h,x:x+w]==i
            ink=float((1-p[y:y+h,x:x+w][component].astype(np.float64)).mean())
            assert abs(ink-c['docres_mean_ink_probability'])<1e-7
            for key,t in [('gate_05',.5),('gate_09',.9)]:
                if ink>=t:expected[key][y:y+h,x:x+w]|=component
        for key in expected:assert np.array_equal(expected[key],masks[key])
        gray=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY).astype(np.float64)
        paper=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)))
        ratio=np.minimum(gray/np.maximum(paper,1),1)
        assert np.array_equal(ratio,np.load(src/'scalar_ratio.npy',allow_pickle=False))
        L=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        N=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
        opaper=cv2.morphologyEx(L,cv2.MORPH_CLOSE,kernel)
        npaper=cv2.morphologyEx(N,cv2.MORPH_CLOSE,kernel)
        eligible=roles['protect']&~(roles['hold']|roles['edit']|roles['red']|roles['extra'])
        eligible[:8]=False;eligible[-8:]=False;eligible[:,:8]=False;eligible[:,-8:]=False
        support=eligible&((opaper-L)>=8)
        anchors=support&((npaper-N)>=12)
        _,anchor_labels,anchor_stats,_=cv2.connectedComponentsWithStats(anchors.astype(np.uint8),connectivity=8)
        connectivity={}
        for key in ['ungated','gate_05','gate_09']:
            output=read(src/(key+'_output.png'));mask=masks[key]
            independently_expected=base.copy()
            # Per-channel recomposition instead of the producer's broadcast expression.
            for channel in range(3):
                values=np.rint(base[:,:,channel].astype(np.float64)*ratio).astype(np.uint8)
                independently_expected[:,:,channel][mask]=values[mask]
            assert np.array_equal(output,independently_expected)
            assert np.array_equal(output[~mask],base[~mask])
            keep=roles['red']|roles['hold']|roles['extra']
            assert np.array_equal(output[keep],base[keep])
            assert np.all(output[mask]<=base[mask])
            before=base[mask].astype(np.int32);after=output[mask].astype(np.int32)
            for c1,c2 in [(0,1),(0,2),(1,2)]:
                cross=np.abs(after[:,c1]*before[:,c2]-after[:,c2]*before[:,c1])
                assert np.all(cross<=(before[:,c1]+before[:,c2])/2), 'Equal colour multiplier up to rounding'
            afterL=cv2.cvtColor(output.astype(np.float32)/255,cv2.COLOR_BGR2LAB)[:,:,0]
            _,afterlabels=cv2.connectedComponents((support&((npaper-afterL)>=12)).astype(np.uint8),connectivity=8)
            counts=dict(selected=0,connected_at_fixed_local_L_contrast_12=0)
            for c in diagnostics['components']:
                x,y,w,h=c['xywh'];i=lookup[tuple(c['xywh']+[c['pixels']])]
                part=(l[y-1:y+h+1,x-1:x+w+1]==i)
                if not mask[y:y+h,x:x+w][l[y:y+h,x:x+w]==i].all():continue
                ring=cv2.dilate(part.astype(np.uint8),np.ones((3,3),np.uint8))>0
                region=anchor_labels[y-1:y+h+1,x-1:x+w+1]
                contact=ring&np.isin(region,c['anchor_components'])
                ids=np.unique(afterlabels[y-1:y+h+1,x-1:x+w+1][contact])
                counts['selected']+=1
                counts['connected_at_fixed_local_L_contrast_12']+=int(len(ids)==1 and ids[0]>0)
            connectivity[key]=counts
        image_keys=['saved_input','saved_npj','ungated_output','gate_09_output']
        for r in spec['regions']:
            x,y,w,h=r['xywh'];canvas=np.asarray(Image.open(REVIEW/(r['id']+'.png')).convert('RGB'))
            scale=min(12,max(1,min(680//w,1000//h)));dw,dh=w*scale,h*scale
            for i,key in enumerate(image_keys):
                a=read(src/(key+'.png'))[y:y+h,x:x+w][:,:,::-1]
                expected_panel=np.asarray(Image.fromarray(a).resize((dw,dh),Image.Resampling.NEAREST))
                left=24+i*(dw+24)
                assert np.array_equal(canvas[120:120+dh,left:left+dw],expected_panel)
                total_panels+=1
        records.append(dict(frame=fid,gate_counts={k:int(v.sum()) for k,v in masks.items()},
            fixed_contrast_connectivity=connectivity,colour_channel_multiplier_exact_up_to_rounding=True,
            outside_mask_and_red_hold_extra_exact=True))
    assert total_panels==80
    dump(OUT/'verification.json',dict(status='passed',frames=6,native_ROI_panels=total_panels,
        independent_component_mean_gate=True,independent_scalar_composition=True,
        colour_rounding_bound=True,input_probability_provenance_matched=True,
        records=records,pixel_GT=False,adopted=False))
    dump(OUT/'manifest.json',dict(dependencies={str(p.relative_to(ROOT)):sha(p)
        for p in [Path(__file__),PILOT/'manifest.json',PILOT/'complete.json',REVIEW/'diagnostics.json']}))
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file()}))
    print(json.dumps(records,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
