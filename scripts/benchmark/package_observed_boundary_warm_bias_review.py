"""Audit reference-only medians and bounded corrections independently."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import probe_observed_boundary_warm_bias as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle
from package_paper_boundary_reference_bias_review import adjacency

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/observed_boundary_warm_bias_review_20261001'


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    rows=json.loads((prior.OUT/'diagnostics.json').read_text())
    originals={r['page']['page']:r['page']['original'] for r in prior.diagnostic.previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    originals.update({r['page']:r['original'] for r in json.loads((prior.diagnostic.fresh.INPUT/'baseline_review.json').read_text())['cases']})
    checks=[];unchanged={mode:[] for mode in prior.MODES}
    for key,path in originals.items():
        original=read(ROOT/path)
        for row in [r for r in rows if r['page']==key]:
            rid=row['id'];native=prior.OUT/'native'/rid;source=ROOT/row['local'];x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh']
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w));crop=original[y:y+h,x:x+w]
            base=read(native/'baseline.png');app=read(native/'applied_context.png',True);known=read(native/'reference_context.png',True)
            assert np.array_equal(base,read(source/row['output_file'])) and np.array_equal(read(native/'original.png'),crop)
            assert np.array_equal(app,read(source/row['applied_file'],True)) and np.array_equal(known,read(source/'reference_context.png',True))
            if app[s].any():
                lab=cv2.cvtColor(original[cs].astype(np.float32)/255,cv2.COLOR_BGR2LAB)
                field=np.load(native/'baseline_field_Lab.npy',allow_pickle=False)
                assert np.array_equal(field,np.load(ROOT/row['field_source'],allow_pickle=False))
                mass=cv2.GaussianBlur(known.astype(np.float32),(0,0),32)
                guide=cv2.GaussianBlur(np.where(known[:,:,None],lab,0),(0,0),32)/np.maximum(mass[:,:,None],1e-6)
                assert np.array_equal(mass,np.load(native/'reference_mass.npy',allow_pickle=False))
                assert np.array_equal(guide,np.load(native/'guide_Lab.npy',allow_pickle=False))
                residual=lab-guide;supported=known&(mass>=.05)
                distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);fade=np.clip((distance-1)/8.,0,1)*app
                assert np.array_equal(fade,np.load(native/'fade.npy',allow_pickle=False))
                n,labels,stats,_=cv2.connectedComponentsWithStats(app.astype(np.uint8),connectivity=4)
                assert len(row['warm_bias_components'])==n-1
                correction={mode:np.zeros_like(field) for mode in prior.MODES}
                for note in row['warm_bias_components']:
                    k=note['id'];component=labels==k
                    boundary=(adjacency(component)>0)&supported
                    ring=supported&(cv2.distanceTransform((~component).astype(np.uint8),cv2.DIST_L2,5)<=64)&~boundary
                    assert note['xywh']==stats[k,:4].tolist() and note['area']==int(stats[k,4])
                    assert note['boundary']==int(boundary.sum()) and note['ring']==int(ring.sum())
                    b=np.median(residual[boundary],axis=0) if boundary.any() else None;r=np.median(residual[ring],axis=0) if ring.any() else None
                    bias=b-r if b is not None and r is not None else None
                    for value,name in [(b,'boundary_median'),(r,'ring_median'),(bias,'bias')]:
                        if value is None:assert note[name] is None
                        else:assert np.array_equal(value,np.array(note[name],np.float32))
                    accepted=bool(boundary.sum()>=16 and ring.sum()>=256 and b[2]>=1 and bias[2]>=1 and b[1]>=0)
                    assert accepted==note['accepted']
                    offset=np.array([min(3,max(0,-float(bias[0]))),min(0,max(-1.5,-float(bias[1]))),min(0,max(-1.5,-float(bias[2])))],np.float32) if accepted else np.zeros(3,np.float32)
                    assert np.array_equal(offset,np.array(note['offset_Lab'],np.float32))
                    for mode in prior.MODES:
                        delta=offset.copy()
                        if mode=='b_only':delta[:2]=0
                        for channel in range(3):correction[mode][:,:,channel][component]=fade[component]*delta[channel]
                for mode in prior.MODES:
                    md=native/mode;delta=np.load(md/'correction_Lab.npy',allow_pickle=False);assert np.array_equal(delta,correction[mode])
                    updated=np.load(md/'result_field_Lab.npy',allow_pickle=False);assert np.array_equal(updated,field+delta)
                    update=np.any(delta!=0,axis=2)&app;assert np.array_equal(update,read(md/'update_context.png',True))
                    value=read(md/'output.png');expected=base.copy();converted=cv2.cvtColor(updated[s].astype(np.float32),cv2.COLOR_LAB2BGR)
                    converted=np.rint(np.clip(converted,0,1)*255).astype(np.uint8);expected[update[s]]=converted[update[s]]
                    assert np.array_equal(value,expected) and np.array_equal(value[~update[s]],base[~update[s]])
                    assert np.array_equal(value[fade[s]==0],base[fade[s]==0])
                    if mode=='b_only':assert np.array_equal(updated[:,:,:2],field[:,:,:2])
                    assert np.array_equal(value[~app[s]],base[~app[s]])
                    assert row['correction_diagnostics'][mode]['corrected_pixels']==int(update[s].sum())
            else:
                for mode in prior.MODES:assert np.array_equal(read(native/mode/'output.png'),base)
            for mode in prior.MODES:
                value=read(native/mode/'output.png');changed=int(np.any(value!=base,axis=2).sum())
                assert changed==row['correction_diagnostics'][mode]['changed_vs_base']
                if not changed:unchanged[mode].append(rid)
            checks.append(dict(id=rid,**row['correction_diagnostics']))
    assert len(checks)==90
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['unseen_thin_crack','extra_top_branch','extra_second_branch','hole_edge_new','new_margin_hole','extra_brown_stain','new_brown_stamp','unseen_fold']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 境界の暖色偏り補正',
            ['同じ参照・適用・模様から、黄色だけ／暖色Labの補正を比較。','色が変わっても輪郭や平坦面の解決とは数えない。原画素を必ず併記。'])
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,records=checks,unchanged_ROIs=unchanged,
        independent_boundary_and_statistics=True,independent_bounded_correction_reconstruction=True,
        masks_references_applied_baseline_fields_fixed=True,inside_first_boundary_and_outside_update_exact_baseline=True,
        nonreference_poison_invariance_verified_by_pilot=True,pixel_GT=False,new_DDRM=False))
    notes=ROOT/'tmp_work/observed_boundary_warm_bias_review_notes_20261001.md';(OUT/'report.md').write_text(notes.read_text())
    pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==98
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==98
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=98,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered observed boundary warm bias:98 PNGs /90 ROIs',flush=True)


if __name__=='__main__':main()
