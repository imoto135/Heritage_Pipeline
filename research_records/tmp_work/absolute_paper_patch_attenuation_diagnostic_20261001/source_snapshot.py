"""Diagnose patch selection and feather attenuation; do not change restoration."""
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import probe_observed_absolute_paper_patches as prior
from build_npj_v6_comparison import dump,read,sha,write
from check_paper_margin_defect_ddrm import figure
from package_paper_blending_review import check_bundle

ROOT=prior.ROOT
OUT=ROOT/'tmp_work/absolute_paper_patch_attenuation_diagnostic_20261001'
SETTINGS=dict(restoration_changed=False,all90ROIs=True,
    highpass='L minus finite Gaussian9x9 sigma1.2; all input neighbours must be in known/applied respectively',
    donor_highpass='central16x16 from each qualified24x24 raw L patch',
    ring='qualified known neighbours within64px of effective applied, eroded9x9 for finite signal',
    selected='first donor and all placed tile donors, before alpha blending',
    alpha='frozen absolute patch alpha; report ROI applied/core6px quantiles and fractions',
    no_GT=True,development=True)


def summary(values):
    a=np.asarray(values).ravel()
    return dict(n=int(len(a)),median=float(np.median(a)) if len(a) else None,
        mean=float(np.mean(a)) if len(a) else None,std=float(np.std(a)) if len(a) else None,
        quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist() if len(a) else [])


def hp(light):return light-cv2.GaussianBlur(light,(9,9),1.2)


def main():
    OUT.mkdir(parents=True,exist_ok=True);check_bundle(prior.OUT)
    deps={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    for n in ['manifest.json','complete.json']:deps[str((prior.OUT/n).relative_to(ROOT))]=sha(prior.OUT/n)
    manifest=dict(settings=SETTINGS,dependencies=deps,source_sha256=sha(Path(__file__)))
    if (OUT/'manifest.json').exists():assert json.loads((OUT/'manifest.json').read_text())==manifest
    else:dump(OUT/'manifest.json',manifest)
    shutil.copyfile(__file__,OUT/'source_snapshot.py');rows=json.loads((prior.OUT/'diagnostics.json').read_text());results=[];current=None
    for row in rows:
        key=row['page'];rid=row['id'];native=prior.OUT/'native'/rid
        if key!=current:
            original=read(ROOT/row['original_source']);full_lab=cv2.cvtColor(original.astype(np.float32)/255,cv2.COLOR_BGR2LAB);current=key
        cx,cy,cw,ch=row['context_xywh'];x,y,w,h=row['xywh'];cs=(slice(cy,cy+ch),slice(cx,cx+cw));s=(slice(y-cy,y-cy+h),slice(x-cx,x-cx+w))
        lab=full_lab[cs];known=read(native/'reference_context.png',True);app=read(native/'applied_context.png',True)
        dest=OUT/'native'/rid;dest.mkdir(parents=True,exist_ok=True)
        finite_known=cv2.erode(known.astype(np.uint8),np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        distance_to_app=cv2.distanceTransform((~app).astype(np.uint8),cv2.DIST_L2,5);ring=finite_known&(distance_to_app<=64)&~app
        rawhp=hp(lab[:,:,0]);ringstats=summary(rawhp[ring])
        positions=json.loads((native/'donors.json').read_text());patch_stats=[];selected=[];first=[];alphastats=None;fieldstats={};corestats=None;poison=True
        if app[s].any():
            for px,py in positions['patch_xy']:
                patch=lab[py:py+24,px:px+24,0];patch_stats.append(float(np.std(hp(patch)[4:20,4:20])))
            for component in positions['components']:
                ids=[r['donor'] for r in component['tiles']];selected.extend(patch_stats[i] for i in ids)
                first.append(dict(component=component['id'],first_donor=ids[0],first_std=patch_stats[ids[0]],all_tiles=len(ids)))
            alpha=np.load(native/'alpha.npy',allow_pickle=False);dist=cv2.distanceTransform(app.astype(np.uint8),cv2.DIST_L2,5);core=app[s]&(dist[s]>=6)
            a=alpha[s][app[s]];alphastats={**summary(a),**{f'fraction_lt_{t}':float(np.mean(a<t)) for t in [.1,.5,.9]}}
            corestats=summary(alpha[s][core])
            finite_app=cv2.erode(app.astype(np.uint8),np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
            roi_mask=np.zeros_like(app);roi_mask[s]=True;finite_app&=roi_mask
            for name,file in [('baseline','baseline_field_Lab.npy'),('unblended_quilt','quilt_Lab.npy'),('observed_L','observed_L_field_Lab.npy'),('observed_Lab','observed_Lab_field_Lab.npy')]:
                field=np.load(native/file,allow_pickle=False);fieldstats[name]=summary(hp(field[:,:,0])[finite_app])
            bad=lab[:,:,0].copy();bad[~known]=230
            poison=np.array_equal(hp(bad)[ring],rawhp[ring]);assert poison
            write(dest/'alpha_ROI.png',np.rint(alpha[s]*255).astype(np.uint8));write(dest/'finite_applied_context.png',finite_app)
        else:write(dest/'alpha_ROI.png',np.zeros((h,w),np.uint8))
        write(dest/'finite_known_ring_context.png',ring)
        dump(dest/'patch_highpass_std.json',patch_stats)
        record={**row,'attenuation':dict(known_ring=ringstats,candidate_patch_stds=summary(patch_stats),selected_tile_patch_stds=summary(selected),
            first_patches=first,alpha_applied=alphastats,alpha_core6=corestats,field_highpass_finite=fieldstats,known_only_poison_invariance=poison)}
        results.append(record)
        if app[s].any():
            original_crop=read(native/'original.png');base=read(native/'baseline.png');value=read(native/'observed_Lab.png');alpha_image=read(dest/'alpha_ROI.png')
            # read() loads the scalar PNG as BGR for a comparable image panel.
            figure([original_crop,base,value,alpha_image],['原画像','現行補修','観測Labパッチ','元パッチ移行重み・白1'],OUT/'comparisons'/(rid+'.png'),row['title']+' · 模様選択と混合減衰',
                f"候補HPstd中央値{np.median(patch_stats):.3f}" if patch_stats else '資格24px候補なし・基準保持',
                '復元変更なし。原模様は推定できない。白い重みを復元成功や正解範囲と読まない。')
        dump(OUT/'progress.json',dict(completed=len(results),total=90,latest=rid));print('Patch attenuation:',len(results),'/90',rid,flush=True)
    dump(OUT/'diagnostics.json',results)
    dump(OUT/'verification.json',dict(status='passed',restoration_changed=False,ROIs=90,finite_kernel_poison_invariance=True))
    for p,h in deps.items():assert sha(ROOT/p)==h
    dump(OUT/'complete.json',dict(outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in ['manifest.json','complete.json']}))
    print('Complete: absolute paper patch selection/attenuation diagnostic',flush=True)


if __name__=='__main__':main()
