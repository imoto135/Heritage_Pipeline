"""Independently audit virtual support and colour-only physical updates."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import probe_paper_colour_virtual_reference_gap as prior
from build_npj_v6_comparison import dump,read,sha
from package_known_paper_fill_review import verify_field
from package_raw_qualified_paper_colour_review import support_check
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/paper_colour_virtual_reference_gap_review_20261001'


def diagnostic(image,base,app,known,s):
    lab=cv2.cvtColor(image.astype(np.float32)/255,cv2.COLOR_BGR2LAB);blab=cv2.cvtColor(base.astype(np.float32)/255,cv2.COLOR_BGR2LAB)
    target=app[s];reference=known[s];jumps=[]
    for ta,tb,ra,rb,diff in [(target[:-1],target[1:],reference[:-1],reference[1:],lab[:-1]-lab[1:]),
        (target[:,:-1],target[:,1:],reference[:,:-1],reference[:,1:],lab[:,:-1]-lab[:,1:])]:
        selected=(ta&rb)|(tb&ra)
        if selected.any():jumps.append(np.linalg.norm(diff[selected],axis=1))
    values=np.concatenate(jumps) if jumps else np.empty(0)
    return dict(boundary_pairs=len(values),median_boundary_deltaLab=float(np.median(values)) if len(values) else None,
        mean_ROI_colour_change_Lab=np.mean((lab-blab)[target],axis=0).tolist() if target.any() else None)


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT);check_bundle(prior.prior.diagnostic.previous.texture.OUT)
    rows=json.loads((prior.OUT/'diagnostics.json').read_text());checked=[];residuals=[];unchanged={m:[] for m in prior.MODES};current=None
    for row in rows:
        key=row['page'];rid=row['id'];native=prior.OUT/'native'/rid;source=prior.prior.OUT/'native'/rid
        if key!=current:
            original=read(ROOT/row['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);current=key
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=full_lab[cs];base=read(native/'baseline.png');app=read(native/'applied_context.png',True);known=read(native/'texture_reference_context.png',True)
        assert np.array_equal(base,read(source/'baseline.png')) and np.array_equal(app,read(source/'applied_context.png',True)) and np.array_equal(known,read(source/'reference_context.png',True))
        assert np.array_equal(read(native/'original.png'),original[y:y+h,x:x+w]);outside_distance=cv2.distanceTransform((~app).astype(np.uint8),cv2.DIST_L2,5)
        if app[s].any():
            field=np.load(native/'baseline_field_Lab.npy',allow_pickle=False);old=np.load(native/'baseline_colour_Lab.npy',allow_pickle=False)
            assert np.array_equal(field,np.load(source/'baseline_field_Lab.npy',allow_pickle=False)) and np.array_equal(old,np.load(ROOT/row['colour_source'],allow_pickle=False))
            weight=np.exp(-(np.maximum(cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5)-1,0)/8.)**2)*app
        for mode,radius in prior.MODES.items():
            md=native/mode;virtual=read(md/'virtual_context.png',True);reference=read(md/'colour_reference_context.png',True)
            accepted=read(md/'virtual_accepted_context.png',True);actual=read(md/'physical_supported_context.png',True);used=read(md/'colour_used_context.png',True)
            update=read(md/'update_context.png',True);fallback=read(md/'fallback_context.png',True);value=read(md/'output.png')
            assert np.array_equal(virtual,app|(known&(outside_distance<=radius))) and np.array_equal(reference,known&~virtual)
            assert np.array_equal(virtual&~app,known&~reference) and not np.any(reference&virtual)
            assert np.array_equal(actual,accepted&app) and np.array_equal(fallback,app&~actual)
            support=[]
            if app[s].any():
                support=support_check(virtual,reference,accepted);fields=np.load(md/'harmonic_fields_Lab.npz',allow_pickle=False)
                mass=cv2.GaussianBlur(reference.astype(np.float32),(0,0),6)
                smooth=cv2.GaussianBlur(np.where(reference[:,:,None],lab,0),(0,0),6)/np.maximum(mass[:,:,None],1e-6)
                for name,target in [('boundary_raw',lab),('boundary_smooth',smooth)]:
                    error,expected_used=verify_field(fields[name],accepted,reference,target);assert np.array_equal(expected_used,used)
                    residuals.append(dict(id=rid,mode=mode,field=name,max_residual=error))
                colour=fields['boundary_smooth']+(fields['boundary_raw']-fields['boundary_smooth'])*weight[:,:,None]
                assert np.array_equal(colour,np.load(md/'colour_field_Lab.npy',allow_pickle=False))
                delta=np.zeros_like(field);delta[actual]=colour[actual]-old[actual]
                assert np.array_equal(delta,np.load(md/'colour_delta_Lab.npy',allow_pickle=False))
                updated=field+delta;assert np.array_equal(updated,np.load(md/'result_field_Lab.npy',allow_pickle=False))
                assert np.array_equal(update,actual&np.any(delta!=0,axis=2))
                expected=base.copy();converted=cv2.cvtColor(updated[s].astype(np.float32),cv2.COLOR_LAB2BGR)
                converted=np.rint(np.clip(converted,0,1)*255).astype(np.uint8);expected[update[s]]=converted[update[s]]
                assert np.array_equal(expected,value)
                # Removing the saved fine L and new colour recovers the old colour, to float precision.
                texture=field[:,:,0]-old[:,:,0]
                assert np.allclose(updated[:,:,0][actual]-colour[:,:,0][actual],texture[actual],atol=2e-5,rtol=0)
            else:assert not accepted.any() and not actual.any() and not update.any() and np.array_equal(value,base)
            assert np.array_equal(value[~actual[s]],base[~actual[s]]) and np.array_equal(value[~app[s]],base[~app[s]])
            stat=row['virtual_gap_diagnostics'][mode];changed=int(np.any(value!=base,axis=2).sum())
            assert changed==stat['changed_vs_base'] and stat['fallback_ROI']==int(fallback[s].sum()) and stat['physical_supported_ROI']==int(actual[s].sum())
            assert stat['virtual_extra_context']==int((virtual&~app).sum()) and stat['removed_colour_reference']==int((known&~reference).sum())
            if not changed:unchanged[mode].append(rid)
            checked.append(dict(id=rid,mode=mode,**stat,support=support,output_colour_diagnostic=diagnostic(value,base,app,known,s)))
        assert np.array_equal(read(native/'control0/output.png'),base)
        print('Audit virtual colour:',len(checked)//3,'/90',rid,flush=True)
    assert len(checked)==270 and len(unchanged['control0'])==90
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['unseen_thin_crack','extra_top_branch','extra_second_branch','hole_edge_new','new_margin_hole','extra_brown_stain','new_brown_stamp','unseen_fold']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 紙色の隣接参照を保留',
            ['仮領域の画素は数値場だけ。実際の編集・元細模様は不変。','マスク外の暖色縁は残す。色面が薄くなることと復元成功を分ける。'])
    dump(OUT/'verification.json',dict(status='passed',pages=29,ROIs=90,conditions=270,records=checked,residuals=residuals,unchanged_ROIs=unchanged,
        virtual_masks_and_reference_subsets_reproduced=True,independent_virtual_component_support_and_direct_residual=True,
        physical_updated_subset_old_applied=True,colour_delta_and_compositing_exact=True,original_fine_L_preserved=True,
        control0_all90_exact_baseline=True,new_DDRM=False,pixel_GT=False))
    notes=ROOT/'tmp_work/paper_colour_virtual_reference_gap_review_notes_20261001.md'
    if not notes.exists():print('Virtual colour audit ready; create notes',flush=True);return
    (OUT/'report.md').write_text(notes.read_text());pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==98
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==98
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=98,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered virtual colour-reference gap:98 PNGs /270 conditions',flush=True)


if __name__=='__main__':main()
