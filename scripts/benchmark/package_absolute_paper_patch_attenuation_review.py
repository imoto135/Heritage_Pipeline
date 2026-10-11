"""Audit finite-support texture summaries and alpha without changing outputs."""
import json
from pathlib import Path
import shutil
import zipfile
import cv2
import numpy as np
import diagnose_absolute_paper_patch_attenuation as prior
from build_npj_v6_comparison import dump,read,sha
from package_defect_context_review import stack
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/absolute_paper_patch_attenuation_review_20261001'


def hp(L):
    kernel=cv2.getGaussianKernel(9,1.2,cv2.CV_64F)
    return L-cv2.sepFilter2D(L,cv2.CV_32F,kernel,kernel)


def stats(values):
    a=np.asarray(values).ravel()
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,mean=float(np.mean(a)) if len(a) else None,
        std=float(np.std(a)) if len(a) else None,quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else [])


def compare(a,b):
    assert a.keys()==b.keys()
    for key in a:
        if a[key] is None:assert b[key] is None
        else:assert np.allclose(a[key],b[key],atol=5e-5,rtol=0),(key,a[key],b[key])


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT);rows=json.loads((prior.OUT/'diagnostics.json').read_text());checked=[];current=None
    for row in rows:
        key=row['page'];rid=row['id'];native=prior.prior.OUT/'native'/rid
        if key!=current:
            original=read(ROOT/row['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);current=key
        x,y,w,h=row['xywh'];cx,cy,cw,ch=row['context_xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        known=read(native/'reference_context.png',True);app=read(native/'applied_context.png',True);lab=full_lab[cs];a=row['attenuation']
        # Box sums independently certify all81 input positions in each stencil.
        counts=cv2.boxFilter(known.astype(np.float32),-1,(9,9),normalize=False,borderType=cv2.BORDER_CONSTANT)
        finite=counts==81;ring=finite&(cv2.distanceTransform((~app).astype(np.uint8),cv2.DIST_L2,5)<=64)&~app
        assert np.array_equal(ring,read(prior.OUT/'native'/rid/'finite_known_ring_context.png',True));compare(stats(hp(lab[:,:,0])[ring]),a['known_ring'])
        if app[s].any():
            record=json.loads((native/'donors.json').read_text());patchstats=[];selected=[];first=[]
            for px,py in record['patch_xy']:patchstats.append(float(np.std(hp(lab[py:py+24,px:px+24,0])[4:20,4:20])))
            saved=json.loads((prior.OUT/'native'/rid/'patch_highpass_std.json').read_text());assert np.allclose(patchstats,saved,atol=5e-5,rtol=0)
            for comp in record['components']:
                ids=[r['donor'] for r in comp['tiles']];selected.extend(patchstats[i] for i in ids)
                first.append(dict(component=comp['id'],first_donor=ids[0],first_std=patchstats[ids[0]],all_tiles=len(ids)))
            compare(stats(patchstats),a['candidate_patch_stds']);compare(stats(selected),a['selected_tile_patch_stds'])
            assert len(first)==len(a['first_patches'])
            for r,b in zip(first,a['first_patches']):
                assert r['component']==b['component'] and r['first_donor']==b['first_donor'] and r['all_tiles']==b['all_tiles']
                assert abs(r['first_std']-b['first_std'])<5e-5
            alpha=np.load(native/'alpha.npy',allow_pickle=False);dist=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);values=alpha[s][app[s]]
            expected=stats(values);expected.update({f'fraction_lt_{t}':float(np.mean(values<t)) for t in [.1,.5,.9]});compare(expected,a['alpha_applied'])
            compare(stats(alpha[s][app[s]&(dist[s]>=6)]),a['alpha_core6'])
            counts=cv2.boxFilter(app.astype(np.float32),-1,(9,9),normalize=False,borderType=cv2.BORDER_CONSTANT)
            roi=np.zeros_like(app);roi[s]=True;valid=(counts==81)&roi
            assert np.array_equal(valid,read(prior.OUT/'native'/rid/'finite_applied_context.png',True))
            for name,file in [('baseline','baseline_field_Lab.npy'),('unblended_quilt','quilt_Lab.npy'),('observed_L','observed_L_field_Lab.npy'),('observed_Lab','observed_Lab_field_Lab.npy')]:
                field=np.load(native/file,allow_pickle=False);compare(stats(hp(field[:,:,0])[valid]),a['field_highpass_finite'][name])
            assert np.array_equal(np.rint(alpha[s]*255).astype(np.uint8),cv2.imread(str(prior.OUT/'native'/rid/'alpha_ROI.png'),cv2.IMREAD_GRAYSCALE))
        else:
            assert a['alpha_applied'] is None and not a['first_patches'] and not a['field_highpass_finite']
        checked.append(rid)
    (OUT/'comparisons').mkdir(exist_ok=True)
    for p in (prior.OUT/'comparisons').glob('*.png'):shutil.copyfile(p,OUT/'comparisons'/p.name)
    focus=['unseen_thin_crack','extra_top_branch','extra_second_branch','new_margin_hole','hole_edge_new']
    for i,rid in enumerate(focus,1):
        stack([OUT/'comparisons'/(rid+'.png')],OUT/f'{i:02d}_{rid}.png',rid+' · 細模様と混合重み',
            ['測定の全入力画素を参照集合内に制限。出力は変更していない。','候補と選択のstdはパッチ中心16px、場のstdは同じ有限支持内部。'])
    dump(OUT/'verification.json',dict(status='passed',ROIs=90,pages=29,records=checked,finite_stencil_checked_by_independent_box_sums=True,
        highpass_recomputed_with_separable_filters=True,max_absolute_stat_tolerance=5e-5,patch_selection_and_alpha_reproduced=True,restoration_changed=False))
    notes=ROOT/'tmp_work/absolute_paper_patch_attenuation_review_notes_20261001.md'
    if not notes.exists():print('Attenuation audit ready; create notes',flush=True);return
    (OUT/'report.md').write_text(notes.read_text());pngs=sorted(OUT.rglob('*.png'));assert len(pngs)==30
    with zipfile.ZipFile(OUT/'comparison_pngs.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in pngs:z.write(p,str(p.relative_to(OUT)))
    with zipfile.ZipFile(OUT/'comparison_pngs.zip') as z:
        assert len(z.namelist())==30
        for p in pngs:assert z.read(str(p.relative_to(OUT)))==p.read_bytes()
    deps={str(notes.relative_to(ROOT)):sha(notes)}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    dump(OUT/'delivery.json',dict(source_sha256=sha(Path(__file__)),PNG_count=30,dependencies=deps,
        outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery.json'}))
    print('Delivered30 PNGs: paper patch selection/attenuation diagnosis',flush=True)


if __name__=='__main__':main()
