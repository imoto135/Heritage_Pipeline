"""Audit raw qualification, harmonic support, colour shifts, and fallback."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import probe_raw_qualified_paper_colour as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_known_paper_fill_review import verify_field
from package_paper_boundary_reference_bias_review import adjacency
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/raw_qualified_paper_colour_review_20261001'


def support_check(app,known,actual):
    n,labels,stats,_=cv2.connectedComponentsWithStats(app.astype(np.uint8),connectivity=4)
    expected=np.zeros_like(app);records=[]
    for k in range(1,n):
        component=labels==k;pairs=int(adjacency(component)[known].sum())
        accepted=stats[k,4]>=64 and pairs>=16
        if accepted:expected|=component
        records.append(dict(component=k,area=int(stats[k,4]),pairs=pairs,accepted=bool(accepted)))
    assert np.array_equal(expected,actual)
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());checked=[];residuals=[];unchanged={m:[] for m in prior.MODES}
    originals={r['page']['page']:r['page']['original'] for r in prior.diagnostic.previous.texture.history.mask.diagnostic.prior.mask.pale.prior.descriptors()}
    originals.update({r['page']:r['original'] for r in json.loads((prior.diagnostic.fresh.INPUT/'baseline_review.json').read_text())['cases']})
    estimate=prior.diagnostic.previous.diagnostic.references.prior.mask.pale.prior.anchor.prior.mask.prior.MARGIN.first.paper_estimate
    for key,path in originals.items():
        original=read(ROOT/path);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
        median=cv2.cvtColor(cv2.medianBlur(original,3).astype(np.float32)/255,cv2.COLOR_BGR2LAB);diff=full_lab-estimate(median)
        rawL=(full_lab[:,:,0]>=50)&(diff[:,:,0]>=-4)&(diff[:,:,0]<=6)
        rawLab=rawL&(np.abs(diff[:,:,1])<=2)&(np.abs(diff[:,:,2])<=4)
        assert np.array_equal(rawL,read(prior.OUT/'pages'/key/'raw_L_ok.png',True))
        assert np.array_equal(rawLab,read(prior.OUT/'pages'/key/'raw_Lab_ok.png',True))
        for row in [r for r in rows if r['page']==key]:
            rid=row['id'];native=prior.OUT/'native'/rid;source=ROOT/row['local'];x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh']
            cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w));crop=original[y:y+h,x:x+w];lab=full_lab[cs]
            base=read(native/'baseline.png');app=read(native/'applied_context.png',True);known=read(native/'texture_reference_context.png',True)
            assert np.array_equal(base,read(source/row['output_file'])) and np.array_equal(read(native/'original.png'),crop)
            assert np.array_equal(app,read(source/row['applied_file'],True)) and np.array_equal(known,read(source/'reference_context.png',True))
            expected_refs=dict(recomputed_colour=known,raw_L_colour=known&rawL[cs],raw_Lab_colour=known&rawLab[cs])
            if app[s].any():
                field=np.load(native/'baseline_field_Lab.npy',allow_pickle=False);colour=np.load(native/'baseline_color_Lab.npy',allow_pickle=False)
                assert np.array_equal(field,np.load(ROOT/row['field_source'],allow_pickle=False))
                assert np.array_equal(colour,np.load(ROOT/row['colour_source'],allow_pickle=False))
                distance=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);weight=np.exp(-(np.maximum(distance-1,0)/8.)**2)*app
            for mode in prior.MODES:
                md=native/mode;ref=read(md/'colour_reference_context.png',True);actual=read(md/'colour_supported_context.png',True)
                used=read(md/'colour_used_context.png',True);update=read(md/'update_context.png',True);fallback=read(md/'fallback_context.png',True);value=read(md/'output.png')
                assert np.array_equal(ref,expected_refs[mode]) and np.array_equal(fallback,app&~actual)
                assert not np.any(actual&~app) and not np.any(ref&app)
                support=support_check(app,ref,actual) if app[s].any() else []
                if app[s].any():
                    fields=np.load(md/'harmonic_fields_Lab.npz',allow_pickle=False)
                    mass=cv2.GaussianBlur(ref.astype(np.float32),(0,0),6)
                    smooth=cv2.GaussianBlur(np.where(ref[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                    for name,target in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                        error,observed_used=verify_field(fields[name],actual,ref,target);assert np.array_equal(observed_used,used)
                        residuals.append(dict(id=rid,mode=mode,field=name,max_residual=error))
                    newcolour=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                    assert np.array_equal(newcolour,np.load(md/'colour_field_Lab.npy',allow_pickle=False))
                    delta=np.zeros_like(field);delta[actual]=newcolour[actual]-colour[actual]
                    assert np.array_equal(delta,np.load(md/'colour_delta_Lab.npy',allow_pickle=False))
                    updated=field+delta;assert np.array_equal(updated,np.load(md/'result_field_Lab.npy',allow_pickle=False))
                    assert np.array_equal(update,actual&np.any(delta!=0,axis=2))
                    expected=base.copy();converted=cv2.cvtColor(updated[s].astype(np.float32),cv2.COLOR_LAB2BGR)
                    converted=np.rint(np.clip(converted,0,1)*255).astype(np.uint8);expected[update[s]]=converted[update[s]]
                    assert np.array_equal(value,expected)
                else:assert not actual.any() and not update.any() and np.array_equal(value,base)
                assert np.array_equal(value[~actual[s]],base[~actual[s]])
                stat=row['colour_diagnostics'][mode];changed=int(np.any(value!=base,axis=2).sum())
                assert stat['changed_vs_base']==changed and stat['supported_ROI']==int(actual[s].sum()) and stat['fallback_ROI']==int(fallback[s].sum())
                assert stat['colour_reference']==int(ref.sum()) and stat['removed_reference']==int((known&~ref).sum())
                assert stat['changed_vs_recomputed_colour']==int(np.any(value!=read(native/'recomputed_colour/output.png'),axis=2).sum())
                if not changed:unchanged[mode].append(rid)
                checked.append(dict(id=rid,mode=mode,**stat,support=support))
            assert np.array_equal(read(native/'recomputed_colour/output.png'),base)
    assert len(checked)==270 and len(unchanged['recomputed_colour'])==90
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['unseen_thin_crack','extra_top_branch','extra_second_branch','hole_edge_new','new_margin_hole','extra_brown_stain','new_brown_stamp','brown_false_red_new','unseen_fold']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 生画素でも資格を満たす紙色参照',
            ['再計算対照・生L資格・生Lab資格。細かい元L模様は固定。','不支持成分は前回を保持。マスク追加なし。色変更を復元成功と数えない。'])
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,conditions=270,records=checked,residuals=residuals,
        unchanged_ROIs=unchanged,recomputed_colour_all_outputs_exact_baseline=True,
        raw_gates_and_nested_colour_references_reproduced=True,independent_pair_and_support_check=True,
        colour_delta_and_composition_reconstructed=True,texture_reference_and_original_source_fixed=True,
        fallback_and_outside_colour_support_exact_baseline=True,new_DDRM=False,pixel_GT=False))
    notes=ROOT/'tmp_work/raw_qualified_paper_colour_review_notes_20261001.md';(OUT/'report.md').write_text(notes.read_text())
    pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==99
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==99
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=99,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered raw-qualified paper colour:99 PNGs /270 conditions',flush=True)


if __name__=='__main__':main()
